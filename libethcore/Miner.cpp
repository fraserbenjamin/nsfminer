
#include "Miner.h"
#include "Farm.h"

namespace dev
{
namespace eth
{

FarmFace* FarmFace::m_this = nullptr;

DeviceDescriptor Miner::getDescriptor()
{
    return m_deviceDescriptor;
}

void Miner::setWork(WorkPackage const& _work)
{
    {
        lock_guard<mutex> l(miner_work_mutex);

        // Void work if this miner is paused
        if (paused())
            m_work.header = h256();
        else
            m_work = _work;
#ifdef DEV_BUILD
        m_workSwitchStart = chrono::steady_clock::now();
#endif
    }
    kick_miner();
}

void Miner::ReportSolution(const h256& header, uint64_t nonce)
{
    cnote << EthWhite << "Job: " << header.abridged()
          << " Solution: " << toHex(nonce, HexPrefix::Add);
}

void Miner::ReportDAGDone(uint64_t dagSize, uint32_t dagTime)
{
    cnote << dev::getFormattedMemory(float(dagSize)) << " of DAG data generated in " << fixed
          << setprecision(1) << dagTime / 1000.0f << " seconds";
}

void Miner::ReportGPUMemoryUsage(uint64_t requiredTotalMemory, uint64_t totalMemory)
{
    cnote << "Using " << dev::getFormattedMemory(float(requiredTotalMemory)) << " out of "
          << dev::getFormattedMemory(float(totalMemory)) << " GPU memory";
}

void Miner::ReportGPUNoMemoryAndPause(uint64_t requiredMemory, uint64_t totalMemory)
{
    cwarn << "Epoch " << m_epochContext.epochNumber << " requires "
          << dev::getFormattedMemory((double)requiredMemory) << " memory. Only "
          << dev::getFormattedMemory((double)totalMemory) << " available on device.";
    pause(MinerPauseEnum::PauseDueToInsufficientMemory);
}

void Miner::pause(MinerPauseEnum what) 
{
    lock_guard<mutex> l(x_pause);
    m_pauseFlags.set(what);
    m_work.header = h256();
    kick_miner();
}

bool Miner::paused()
{
    lock_guard<mutex> l(x_pause);
    return m_pauseFlags.any();
}

bool Miner::pauseTest(MinerPauseEnum what)
{
    lock_guard<mutex> l(x_pause);
    return m_pauseFlags.test(what);
}

string Miner::pausedString()
{
    lock_guard<mutex> l(x_pause);
    string retVar;
    if (m_pauseFlags.any())
    {
        for (int i = 0; i < MinerPauseEnum::Pause_MAX; i++)
        {
            if (m_pauseFlags[(MinerPauseEnum)i])
            {
                if (!retVar.empty())
                    retVar.append("; ");

                if (i == MinerPauseEnum::PauseDueToOverHeating)
                    retVar.append("Overheating");
                else if (i == MinerPauseEnum::PauseDueToAPIRequest)
                    retVar.append("Api request");
                else if (i == MinerPauseEnum::PauseDueToFarmPaused)
                    retVar.append("Farm suspended");
                else if (i == MinerPauseEnum::PauseDueToInsufficientMemory)
                    retVar.append("Insufficient GPU memory");
                else if (i == MinerPauseEnum::PauseDueToInitEpochError)
                    retVar.append("Epoch initialization error");

            }
        }
    }
    return retVar;
}

void Miner::resume(MinerPauseEnum fromwhat) 
{
    lock_guard<mutex> l(x_pause);
    m_pauseFlags.reset(fromwhat);
    //if (!m_pauseFlags.any())
    //{
    //    // TODO Push most recent job from farm ?
    //    // If we do not push a new job the miner will stay idle
    //    // till a new job arrives
    //}
}

float Miner::RetrieveHashRate() noexcept
{
    return m_hashRate.load(memory_order_relaxed);
}

void Miner::TriggerHashRateUpdate() noexcept
{
    bool b = false;
    if (m_hashRateUpdate.compare_exchange_weak(b, true))
        return;
    // GPU didn't respond to last trigger, assume it's dead.
    // This can happen on CUDA if:
    //   runtime of --cuda-grid-size * --cuda-streams exceeds time of m_collectInterval
    m_hashRate = 0.0;
}

WorkPackage Miner::work() const
{
    unique_lock<mutex> l(miner_work_mutex);
    return m_work;
}

void Miner::updateHashRate(uint32_t _groupSize, uint32_t _increment) noexcept
{
    m_groupCount += _increment * _groupSize;
    bool b = true;
    if (!m_hashRateUpdate.compare_exchange_weak(b, false))
        return;
    using namespace chrono;
    auto t = steady_clock::now();
    auto us = duration_cast<microseconds>(t - m_hashTime).count();
    m_hashTime = t;

    m_hashRate.store(us ? (float(m_groupCount) * 1.0e6f) / us : 0.0f, memory_order_relaxed);
    m_groupCount = 0;
}

void Miner::workLoop()
{
    WorkPackage current;
    current.header = h256();

    if (!initDevice())
        return;

    try
    {
        while (!shouldStop())
        {
            const WorkPackage w = work();
            if (!w)
            {
                m_hung_miner.store(false);
                unique_lock<mutex> l(miner_work_mutex);
                m_new_work_signal.wait_for(l, chrono::seconds(3));
                continue;
            }

            // Epoch change ?
            if (current.epoch != w.epoch)
            {
                if (!initEpoch())
                    break;

                // As DAG generation takes a while we need to
                // ensure we're on latest job, not on the one
                // which triggered the epoch change
                current = w;
                continue;
            }

            // Persist most recent job.
            // Job's differences should be handled at higher level
            current = w;

            uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)current.boundary >> 192);

            miner_adjust_work_multiple();

            // Eventually start searching
            search(current.header, upper64OfBoundary, current.startNonce, w);
        }

        // Reset miner and stop working
        miner_reset_device();
    }
    catch (const runtime_error& e)
    {
        throw runtime_error(string("GPU error: ") + e.what());
    }
}

void Miner::search(
    const h256& header, uint64_t target, uint64_t start_nonce, const dev::eth::WorkPackage& w)
{
    miner_set_header(header);
    if (m_current_target != target)
    {
        miner_set_target(target);
        m_current_target = target;
    }
    const uint32_t streams(miner_get_streams());
    const uint32_t stream_blocks(miner_get_stream_blocks());
    const uint32_t batch_blocks(streams * stream_blocks);

    // NOTE: The following struct must match the one defined in
    // ethash.cl
    struct SearchResults search_buf[MAX_STREAMS];

    // prime each stream, clear search result buffers and start the search
    for (uint32_t streamIdx = 0; streamIdx < streams; streamIdx++, start_nonce += batch_blocks)
    {
        miner_clear_counts(streamIdx);
        m_hung_miner.store(false);
        miner_search(streamIdx, search_buf[streamIdx], start_nonce);
    }

    bool done(false);
    uint32_t streams_bsy((1 << streams) - 1);

    // process stream batches until we get new work.

    while (streams_bsy)
    {
        if (!done)
            done = paused();

        uint32_t batchCount(0);

        // This inner loop will process each cuda stream individually
        for (uint32_t streamIdx = 0; streamIdx < streams; streamIdx++, start_nonce += batch_blocks)
        {
            uint32_t stream_mask(1 << streamIdx);
            if (!(streams_bsy & stream_mask))
                continue;

            struct SearchResults& r(search_buf[streamIdx]);

            // Wait for the stream complete
            miner_sync(streamIdx);

            // clear solution count, hash count and done
            miner_clear_counts(streamIdx);

            if (r.count > MAX_RESULTS)
                r.count = MAX_RESULTS;
            batchCount += r.hashCount;

            if (done)
                streams_bsy &= ~stream_mask;
            else
            {
                m_hung_miner.store(false);
                miner_search(streamIdx, search_buf[streamIdx], start_nonce);
            }

            if (r.count)
                for (uint32_t i = 0; i < r.count; i++)
                {
                    uint64_t nonce(start_nonce - stream_blocks + r.rslt[i].gid);
                    h256 mix((::byte*)&r.rslt[i].mix, h256::ConstructFromPointer);

                    Farm::f().submitProof(
                        Solution{nonce, mix, w, chrono::steady_clock::now(), m_index});
                    ReportSolution(w.header, nonce);
                }
            if (shouldStop())
                done = true;
        }
        updateHashRate(m_deviceDescriptor.cuBlockSize, batchCount);
    }

#ifdef DEV_BUILD
    // Optionally log job switch time
    if (!shouldStop() && (g_logOptions & LOG_SWITCH))
        cnote << "Switch time: "
              << chrono::duration_cast<chrono::microseconds>(
                     chrono::steady_clock::now() - m_workSwitchStart)
                     .count()
              << " us.";
#endif
}

}  // namespace eth
}  // namespace dev
