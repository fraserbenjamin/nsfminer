
#include <boost/dll.hpp>

#include <libethcore/Farm.h>
#include <ethash/ethash.hpp>

#include "CLMiner.h"
#include "ethash.h"

using namespace dev;
using namespace eth;

namespace dev
{
namespace eth
{

// WARNING: Do not change the value of the following constant
// unless you are prepared to make the neccessary adjustments
// to the assembly code for the binary kernels.
const size_t c_maxSearchResults = 4;

/**
 * Returns the name of a numerical cl_int error
 * Takes constants from CL/cl.h and returns them in a readable format
 */
static const char* strClError(cl_int err)
{
    switch (err)
    {
    case CL_SUCCESS:
        return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";

#ifdef CL_VERSION_1_2
    case CL_COMPILE_PROGRAM_FAILURE:
        return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
        return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
        return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
        return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif  // CL_VERSION_1_2

    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
        return "CL_INVALID_PROPERTY";

#ifdef CL_VERSION_1_2
    case CL_INVALID_IMAGE_DESCRIPTOR:
        return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
        return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
        return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
        return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif  // CL_VERSION_1_2

#ifdef CL_VERSION_2_0
    case CL_INVALID_PIPE_SIZE:
        return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE:
        return "CL_INVALID_DEVICE_QUEUE";
#endif  // CL_VERSION_2_0

#ifdef CL_VERSION_2_2
    case CL_INVALID_SPEC_ID:
        return "CL_INVALID_SPEC_ID";
    case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
        return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif  // CL_VERSION_2_2
    }

    return "Unknown CL error encountered";
}

static string ethCLErrorHelper(const char* msg, cl::Error const& clerr)
{
    ostringstream osstream;
    osstream << msg << ": " << clerr.what() << ": " << strClError(clerr.err()) << " ("
             << clerr.err() << ")";
    return osstream.str();
}

namespace
{
void addDefinition(string& _source, char const* _id, unsigned _value)
{
    char buf[256];
    sprintf(buf, "#define %s %uu\n", _id, _value);
    _source.insert(_source.begin(), buf, buf + strlen(buf));
}

vector<cl::Platform> getPlatforms()
{
    vector<cl::Platform> platforms;
    try
    {
        cl::Platform::get(&platforms);
    }
    catch (cl::Error const& err)
    {
#if defined(CL_PLATFORM_NOT_FOUND_KHR)
        if (err.err() == CL_PLATFORM_NOT_FOUND_KHR)
            cerr << "No OpenCL platforms found" << endl;
        else
#endif
            cerr << "OpenCL error : " << err.what();
    }
    return platforms;
}

vector<cl::Device> getDevices(vector<cl::Platform> const& _platforms, unsigned _platformId)
{
    vector<cl::Device> devices;
    size_t platform_num = min<size_t>(_platformId, _platforms.size() - 1);
    try
    {
        _platforms[platform_num].getDevices(
            CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, &devices);
    }
    catch (cl::Error const& err)
    {
        // if simply no devices found return empty vector
        if (err.err() != CL_DEVICE_NOT_FOUND)
            throw err;
    }
    return devices;
}

}  // namespace

}  // namespace eth
}  // namespace dev

CLMiner::CLMiner(unsigned _index, DeviceDescriptor& _device) : Miner("cl-", _index)
{
    m_deviceDescriptor = _device;
    m_block_multiple = 200000;
}

CLMiner::~CLMiner()
{
    stopWorking();
    miner_kick();
}

void CLMiner::miner_kick()
{
    if (resourceInitialized())
    {
    if (m_abortMutex.try_lock())
    {
        if (m_abortqueue)
        {
            const static uint32_t one = 1;
            m_abortqueue->enqueueWriteBuffer(
                *m_searchBuffer, CL_FALSE, offsetof(Search_results, done), sizeof(one), &one);
        }
        m_abortMutex.unlock();
    }
    }
    m_new_work_signal.notify_one();
}

void CLMiner::enumDevices(map<string, DeviceDescriptor>& _DevicesCollection)
{
    // Load available platforms
    vector<cl::Platform> platforms = getPlatforms();
    if (platforms.empty())
        return;

    unsigned int dIdx = 0;
    for (unsigned int pIdx = 0; pIdx < platforms.size(); pIdx++)
    {
        string platformName = platforms.at(pIdx).getInfo<CL_PLATFORM_NAME>();
        ClPlatformTypeEnum platformType = ClPlatformTypeEnum::Unknown;
        if (platformName == "AMD Accelerated Parallel Processing")
            platformType = ClPlatformTypeEnum::Amd;
        else if (platformName == "Clover" || platformName == "Intel Gen OCL Driver")
            platformType = ClPlatformTypeEnum::Clover;
        else if (platformName == "NVIDIA CUDA")
            platformType = ClPlatformTypeEnum::Nvidia;
        else if (platformName.find("Intel") != string::npos)
            platformType = ClPlatformTypeEnum::Intel;
        else
            continue;

        string platformVersion = platforms.at(pIdx).getInfo<CL_PLATFORM_VERSION>();
        unsigned int platformVersionMajor = stoi(platformVersion.substr(7, 1));
        unsigned int platformVersionMinor = stoi(platformVersion.substr(9, 1));

        dIdx = 0;
        vector<cl::Device> devices = getDevices(platforms, pIdx);
        for (auto const& device : devices)
        {
            DeviceTypeEnum clDeviceType = DeviceTypeEnum::Unknown;
            cl_device_type detectedType = device.getInfo<CL_DEVICE_TYPE>();
            if (detectedType == CL_DEVICE_TYPE_GPU)
                clDeviceType = DeviceTypeEnum::Gpu;
            else if (detectedType == CL_DEVICE_TYPE_CPU)
                clDeviceType = DeviceTypeEnum::Cpu;
            else if (detectedType == CL_DEVICE_TYPE_ACCELERATOR)
                clDeviceType = DeviceTypeEnum::Accelerator;
            else
                continue;

            string uniqueId;
            DeviceDescriptor deviceDescriptor;

            if (clDeviceType == DeviceTypeEnum::Gpu && platformType == ClPlatformTypeEnum::Nvidia)
            {
                cl_int bus_id, slot_id;
                if (clGetDeviceInfo(device.get(), 0x4008 /*CL_DEVICE_PCI_BUS_ID_NV*/,
                        sizeof(bus_id), &bus_id, NULL) == CL_SUCCESS &&
                    clGetDeviceInfo(device.get(), 0x4009 /*CL_DEVICE_PCI_SLOT_ID_NV*/,
                        sizeof(slot_id), &slot_id, NULL) == CL_SUCCESS)
                {
                    ostringstream s;
                    s << setfill('0') << setw(2) << hex << bus_id << ":" << setw(2)
                      << (unsigned int)(slot_id >> 3) << "." << (unsigned int)(slot_id & 0x7);
                    uniqueId = s.str();
                }
            }
            else if (clDeviceType == DeviceTypeEnum::Gpu &&
                     (platformType == ClPlatformTypeEnum::Amd ||
                         platformType == ClPlatformTypeEnum::Clover))
            {
                cl_char t[24];
                if (clGetDeviceInfo(device.get(), 0x4037 /*CL_DEVICE_TOPOLOGY_AMD*/, sizeof(t), &t,
                        NULL) == CL_SUCCESS)
                {
                    ostringstream s;
                    s << setfill('0') << setw(2) << hex << (unsigned int)(t[21]) << ":" << setw(2)
                      << (unsigned int)(t[22]) << "." << (unsigned int)(t[23]);
                    uniqueId = s.str();
                }
            }
            else if (clDeviceType == DeviceTypeEnum::Gpu && platformType == ClPlatformTypeEnum::Intel)
            {
                ostringstream s;
                s << "Intel GPU " << pIdx << "." << dIdx;
                uniqueId = s.str();

            }
            else if (clDeviceType == DeviceTypeEnum::Cpu)
            {
                ostringstream s;
                s << "CPU:" << setfill('0') << setw(2) << hex << (pIdx + dIdx);
                uniqueId = s.str();
            }
            else
            {
                // We're not prepared (yet) to handle other platforms or types
                ++dIdx;
                continue;
            }

            if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
                deviceDescriptor = _DevicesCollection[uniqueId];
            else
                deviceDescriptor = DeviceDescriptor();

            // Fill the blanks by OpenCL means
            deviceDescriptor.name = device.getInfo<CL_DEVICE_NAME>();
            deviceDescriptor.type = clDeviceType;
            deviceDescriptor.uniqueId = uniqueId;
            deviceDescriptor.clDetected = true;
            deviceDescriptor.clPlatformId = pIdx;
            deviceDescriptor.clPlatformName = platformName;
            deviceDescriptor.clPlatformType = platformType;
            deviceDescriptor.clPlatformVersion = platformVersion;
            deviceDescriptor.clPlatformVersionMajor = platformVersionMajor;
            deviceDescriptor.clPlatformVersionMinor = platformVersionMinor;
            deviceDescriptor.clDeviceOrdinal = dIdx;

            deviceDescriptor.clName = deviceDescriptor.name;
            deviceDescriptor.clDeviceVersion = device.getInfo<CL_DEVICE_VERSION>();
            deviceDescriptor.clDeviceVersionMajor =
                stoi(deviceDescriptor.clDeviceVersion.substr(7, 1));
            deviceDescriptor.clDeviceVersionMinor =
                stoi(deviceDescriptor.clDeviceVersion.substr(9, 1));
            deviceDescriptor.totalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
            deviceDescriptor.clGroupSize = 64;

            // Is it an NVIDIA card ?
            if (platformType == ClPlatformTypeEnum::Nvidia)
            {
                size_t siz;
                clGetDeviceInfo(device.get(), CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV,
                    sizeof(deviceDescriptor.clNvComputeMajor), &deviceDescriptor.clNvComputeMajor,
                    &siz);
                clGetDeviceInfo(device.get(), CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV,
                    sizeof(deviceDescriptor.clNvComputeMinor), &deviceDescriptor.clNvComputeMinor,
                    &siz);
                deviceDescriptor.clNvCompute = to_string(deviceDescriptor.clNvComputeMajor) + "." +
                                               to_string(deviceDescriptor.clNvComputeMinor);
            }

            // Upsert Devices Collection
            _DevicesCollection[uniqueId] = deviceDescriptor;
            ++dIdx;

        }
    }

}

bool CLMiner::miner_init_device()
{
    // LookUp device
    // Load available platforms
    vector<cl::Platform> platforms = getPlatforms();
    if (platforms.empty())
        return false;

    vector<cl::Device> devices = getDevices(platforms, m_deviceDescriptor.clPlatformId);
    if (devices.empty())
        return false;

    m_device = devices.at(m_deviceDescriptor.clDeviceOrdinal);

    // Set Hardware Monitor Info
    if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Nvidia)
    {
        m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
        m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
        m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)
    }
    else if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Amd)
    {
        m_hwmoninfo.deviceType = HwMonitorInfoType::AMD;
        m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
        m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)
    }
    else if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Clover)
    {
        m_hwmoninfo.deviceType = HwMonitorInfoType::UNKNOWN;
        m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
        m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)
    }
    else if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Intel)
    {
        m_hwmoninfo.deviceType = HwMonitorInfoType::UNKNOWN;
        m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
        m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)
    }
    else
    {
        // Don't know what to do with this
        cwarn << "Unrecognized Platform";
        return false;
    }

    if (m_deviceDescriptor.clPlatformVersionMajor == 1 &&
        (m_deviceDescriptor.clPlatformVersionMinor == 0 ||
            m_deviceDescriptor.clPlatformVersionMinor == 1))
    {
        if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Clover)
        {
            cwarn
                << "OpenCL " << m_deviceDescriptor.clPlatformVersion
                << " not supported, but platform Clover might work nevertheless. USE AT OWN RISK!";
        }
        else
        {
            cwarn << "OpenCL " << m_deviceDescriptor.clPlatformVersion
                  << " not supported. Minimum required version is 1.2";
            throw new runtime_error("OpenCL 1.2 required");
        }
    }

    ostringstream s;
    s << "Using Pci " << m_deviceDescriptor.uniqueId << ": " << m_deviceDescriptor.clName;

    if (!m_deviceDescriptor.clNvCompute.empty())
        s << " (Compute " + m_deviceDescriptor.clNvCompute + ")";
    else
        s << " (" << m_deviceDescriptor.clDeviceVersion;

    s << ") Memory : " << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory);
    cnote << s.str();

    return true;

}

bool CLMiner::miner_init_epoch()
{
    m_abortMutex.lock();
    auto startInit = chrono::steady_clock::now();
    size_t RequiredMemory = m_epochContext.dagSize + m_epochContext.lightSize;

    // Check whether the current device has sufficient memory every time we recreate the dag
    if (m_deviceDescriptor.totalMemory < RequiredMemory)
    {
        ReportGPUNoMemoryAndPause(RequiredMemory, m_deviceDescriptor.totalMemory);
        return false;  // This will prevent to exit the thread and
                       // Eventually resume mining when changing coin or epoch (NiceHash)
    }

    // Release the pause flag if any
    resume(MinerPauseEnum::PauseDueToInsufficientMemory);
    resume(MinerPauseEnum::PauseDueToInitEpochError);

    ReportGPUMemoryUsage(RequiredMemory, m_deviceDescriptor.totalMemory);

    try
    {

        char options[256] = {0};
        int computeCapability = 0;

        // Nvidia
        if (!m_deviceDescriptor.clNvCompute.empty())
        {
            computeCapability =
                m_deviceDescriptor.clNvComputeMajor * 10 + m_deviceDescriptor.clNvComputeMinor;
            int maxregs = computeCapability >= 35 ? 72 : 63;
            sprintf(options, "-cl-nv-maxrregcount=%d", maxregs);
        }

        free_buffers();
        // create context
        m_context = new cl::Context(vector<cl::Device>(&m_device, &m_device + 1));
        // create new queue with default in order execution property
        m_queue = new cl::CommandQueue(*m_context, m_device);
        m_abortqueue = new cl::CommandQueue(*m_context, m_device);

        m_dagItems = m_epochContext.dagNumItems;

        // patch source code
        // note: The kernels here are simply compiled version of the respective .cl kernels
        // into a byte array by bin2h.cmake. There is no need to load the file by hand in runtime
        // See libethash-cl/CMakeLists.txt: add_custom_command()
        // TODO: Just use C++ raw string literal.
        string code;

        code = string(ethash_cl, ethash_cl + sizeof(ethash_cl));

        addDefinition(code, "WORKSIZE", m_deviceDescriptor.clGroupSize);
        addDefinition(code, "ACCESSES", 64);
        addDefinition(code, "MAX_OUTPUTS", c_maxSearchResults);
        addDefinition(code, "PLATFORM", static_cast<unsigned>(m_deviceDescriptor.clPlatformType));
        addDefinition(code, "COMPUTE", computeCapability);

        // create miner OpenCL program
        cl::Program::Sources sources{{code.data(), code.size()}};
        cl::Program program(*m_context, sources), binaryProgram;
        try
        {
            program.build({m_device}, options);
        }
        catch (cl::BuildError const& buildErr)
        {
            ccrit << "OpenCL kernel build log:\n"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
            ccrit << "OpenCL kernel build error (" << buildErr.err() << "):\n" << buildErr.what();
            pause(MinerPauseEnum::PauseDueToInitEpochError);
            free_buffers();
            return false;
        }

        /* If we have a binary kernel, we load it in tandem with the opencl,
           that way, we can use the dag generate opencl code and fall back on
           the default kernel if loading fails for whatever reason */
        bool loadedBinary = false;
        string device_name = m_deviceDescriptor.clName;
        if (m_deviceDescriptor.clBin)
        {
            ifstream kernel_file;
            vector<unsigned char> bin_data;
            stringstream fname_strm;

            /* Open kernels/ethash_{devicename}_lws{local_work_size}.bin */
            transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
            fname_strm << boost::dll::program_location().parent_path().string()
                       << "/kernels/ethash_" << device_name << "_lws"
                       << m_deviceDescriptor.clGroupSize << ".bin";
            cnote << "Loading binary kernel " << fname_strm.str();
            try
            {
                kernel_file.open(fname_strm.str(), ios::in | ios::binary);

                if (kernel_file.good())
                {
                    /* Load the data vector with file data */
                    kernel_file.unsetf(ios::skipws);
                    bin_data.insert(bin_data.begin(), istream_iterator<unsigned char>(kernel_file),
                        istream_iterator<unsigned char>());

                    /* Setup the program */
                    cl::Program::Binaries blobs({bin_data});
                    cl::Program program(*m_context, {m_device}, blobs);
                    try
                    {
                        program.build({m_device}, options);
                        cnote << "Build info success:"
                              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
                        binaryProgram = program;
                        loadedBinary = true;
                    }
                    catch (cl::Error const&)
                    {
                        cwarn << "Build failed! Info:"
                              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
                        cwarn << fname_strm.str();
                        cwarn << "Falling back to OpenCL kernel...";
                    }
                }
                else
                {
                    cwarn << "Failed to load binary kernel: " << fname_strm.str();
                    cwarn << "Falling back to OpenCL kernel...";
                }
            }
            catch (...)
            {
                cwarn << "Failed to load binary kernel: " << fname_strm.str();
                cwarn << "Falling back to OpenCL kernel...";
            }
        }

        // create buffer for dag
        try
        {
            unsigned delta = (m_epochContext.dagNumItems & 1) ? 64 : 0;
            m_dag[0] =
                new cl::Buffer(*m_context, CL_MEM_READ_ONLY, m_epochContext.dagSize / 2 + delta);
            m_dag[1] =
                new cl::Buffer(*m_context, CL_MEM_READ_ONLY, m_epochContext.dagSize / 2 - delta);
            try
            {
                m_light = new cl::Buffer(*m_context, CL_MEM_READ_ONLY, m_epochContext.lightSize);
            }
            catch (cl::Error const& err)
            {
                if ((err.err() == CL_OUT_OF_RESOURCES) || (err.err() == CL_OUT_OF_HOST_MEMORY))
                {
                    // Ok, no room for light cache on GPU. Try allocating on host
                    ccrit << "No room on GPU";
                    pause(MinerPauseEnum::PauseDueToInitEpochError);
                    free_buffers();
                    throw;
                }
            }

            // If we have a binary kernel to use, let's try it
            // otherwise just do a normal opencl load
            if (loadedBinary)
                m_searchKernel = cl::Kernel(binaryProgram, "search");
            else
                m_searchKernel = cl::Kernel(program, "search");

            m_dagKernel = cl::Kernel(program, "GenerateDAG");

            m_queue->enqueueWriteBuffer(
                *m_light, CL_TRUE, 0, m_epochContext.lightSize, m_epochContext.lightCache);
        }
        catch (cl::Error const& err)
        {
            cwarn << ethCLErrorHelper("Creating DAG buffer failed", err);
            pause(MinerPauseEnum::PauseDueToInitEpochError);
            free_buffers();
            return false;
        }
        // create buffer for header
        m_header = new cl::Buffer(*m_context, CL_MEM_READ_ONLY, 32);

        m_searchKernel.setArg(1, m_header[0]);
        m_searchKernel.setArg(2, *m_dag[0]);
        m_searchKernel.setArg(3, *m_dag[1]);
        m_searchKernel.setArg(4, m_dagItems);

        // create mining buffers
        m_searchBuffer = new cl::Buffer(*m_context, CL_MEM_WRITE_ONLY, sizeof(Search_results));

        m_dagKernel.setArg(1, *m_light);
        m_dagKernel.setArg(2, *m_dag[0]);
        m_dagKernel.setArg(3, *m_dag[1]);
        m_dagKernel.setArg(4, (uint32_t)(m_epochContext.lightSize / 64));

        const uint32_t workItems = m_dagItems * 2;  // GPU computes partial 512-bit DAG items.

        uint32_t start;
        const uint32_t chunk = m_deviceDescriptor.clGroupSize * m_block_multiple;
        for (start = 0; start <= workItems - chunk; start += chunk)
        {
            m_dagKernel.setArg(0, start);
            m_queue->enqueueNDRangeKernel(
                m_dagKernel, cl::NullRange, chunk, m_deviceDescriptor.clGroupSize);
            m_queue->finish();
        }
        if (start < workItems)
        {
            uint32_t groupsLeft = workItems - start;
            groupsLeft =
                (groupsLeft + m_deviceDescriptor.clGroupSize - 1) / m_deviceDescriptor.clGroupSize;
            m_dagKernel.setArg(0, start);
            m_queue->enqueueNDRangeKernel(m_dagKernel, cl::NullRange,
                groupsLeft * m_deviceDescriptor.clGroupSize, m_deviceDescriptor.clGroupSize);
            m_queue->finish();
        }

        auto dagTime =
            chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - startInit);

        ReportDAGDone(m_epochContext.dagSize, uint32_t(dagTime.count()));
    }
    catch (cl::Error const& err)
    {
        ccrit << ethCLErrorHelper("OpenCL init failed", err);
        pause(MinerPauseEnum::PauseDueToInitEpochError);
        free_buffers();
        return false;
    }

    m_searchKernel.setArg(0, *m_searchBuffer);  // Supply output buffer to kernel.
    m_searchKernel.setArg(1, *m_header);
    m_searchKernel.setArg(2, *m_dag[0]);  // Supply DAG buffer to kernel.
    m_searchKernel.setArg(3, *m_dag[1]);  // Supply DAG buffer to kernel.
    m_searchKernel.setArg(4, m_dagItems);

    m_abortMutex.unlock();

#ifdef DEV_BUILD
    if (g_logOptions & LOG_SWITCH)
        cnote << "Switch time: "
              << chrono::duration_cast<chrono::microseconds>(
                     chrono::steady_clock::now() - m_workSwitchStart)
                     .count()
              << " us.";
#endif
    return true;
}

void CLMiner::miner_clear_counts(uint32_t streamIdx)
{
    (void)streamIdx;
    static uint32_t zerox3[3] = {0, 0, 0};
    // clean the solution count, hash count, and abort flag
    m_queue->enqueueWriteBuffer(
        *m_searchBuffer, CL_FALSE, offsetof(Search_results, counts), sizeof(zerox3), zerox3);
}

void CLMiner::miner_reset_device()
{
    // never reset opencl device
}

void CLMiner::miner_search(uint32_t streamIdx, uint64_t start_nonce)
{
    (void)streamIdx;
    // Run the kernel.
    m_searchKernel.setArg(5, start_nonce);
    m_hung_miner.store(false);
    m_queue->enqueueNDRangeKernel(m_searchKernel, cl::NullRange,
        m_deviceDescriptor.clGroupSize * m_block_multiple, m_deviceDescriptor.clGroupSize);
}

void CLMiner::miner_sync(uint32_t streamIdx, Search_results& search_buf)
{
    (void)streamIdx;
    if (m_queue)
    {
        // no need to read the abort flag.
        m_queue->enqueueReadBuffer(*m_searchBuffer, CL_TRUE, offsetof(Search_results, counts),
            2 * sizeof(count_pair), (void*)&search_buf.counts);
        if (search_buf.counts.solCount)
        {
            if (search_buf.counts.solCount > c_maxSearchResults)
                search_buf.counts.solCount = c_maxSearchResults;

            m_queue->enqueueReadBuffer(*m_searchBuffer, CL_TRUE, offsetof(Search_results, results),
                search_buf.counts.solCount * sizeof(Search_Result), (void*)search_buf.results);
        }
    }
    else
        search_buf.counts.solCount = 0;
}

void CLMiner::miner_set_header(const h256& header)
{
    // Update header constant buffer.
    m_queue->enqueueWriteBuffer(*m_header, CL_FALSE, 0, header.size, header.data());
}

void CLMiner::miner_set_target(uint64_t target)
{
    m_searchKernel.setArg(6, target);
}

void CLMiner::miner_get_block_sizes(Block_sizes& blks)
{
    float hr = RetrieveHashRate();
    if (hr > 1e7)
        m_block_multiple = uint32_t(hr * CL_TARGET_BATCH_TIME / m_deviceDescriptor.clGroupSize);
    blks.streams = 1;
    blks.block_size = m_deviceDescriptor.clGroupSize;
    blks.stream_size = m_deviceDescriptor.clGroupSize * m_block_multiple;
}

