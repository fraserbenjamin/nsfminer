
#pragma once

#include "ethash_cuda_miner_kernel.h"

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>

#include <functional>

#define MAX_STREAMS 4
#define CU_TARGET_BATCH_TIME 0.9F  // seconds

namespace dev
{
namespace eth
{
class CUDAMiner : public Miner
{
public:
    CUDAMiner(unsigned _index, DeviceDescriptor& _device);
    ~CUDAMiner() override;

    static int getNumDevices();
    static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);

protected:
    bool initDevice() override;

    bool initEpoch() override;

    void kick_miner() override;

    virtual void miner_set_header(const h256& header) override;
    virtual void miner_set_target(uint64_t _target) override;
    virtual uint32_t miner_get_streams() override;
    virtual uint32_t miner_get_stream_blocks() override;
    virtual void miner_clear_counts(uint32_t streamIdx) override;
    virtual void miner_adjust_work_multiple() override;
    virtual void miner_reset_device() override;
    virtual void miner_search(
        uint32_t streamIdx, SearchResults& search_buf, uint64_t start_nonce) override;
    virtual void miner_sync(uint32_t streamIdx) override;

private:
    void workLoop() override;

    void search(
        uint8_t const* header, uint64_t target, uint64_t _startN, const dev::eth::WorkPackage& w);

    Search_results* m_search_buf[MAX_STREAMS];
    cudaStream_t m_streams[MAX_STREAMS];

    uint64_t m_allocated_memory_dag = 0; // dag_size is a uint64_t in EpochContext struct
    size_t m_allocated_memory_light_cache = 0;

    volatile bool m_done = true;
};


}  // namespace eth
}  // namespace dev
