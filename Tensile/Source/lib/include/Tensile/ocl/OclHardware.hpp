#ifndef OCL_HARDWARE_HPP
#define OCL_HARDWARE_HPP

#include <Tensile/AMDGPU.hpp>
#include <Tensile/Tensile.hpp>

namespace Tensile
{
    namespace ocl
    {
        struct oclDeviceProp_t
        {
            std::string name;
            size_t      totalGlobalMem;
            size_t      sharedMemPerBlock;
            int         warpSize;
            int         maxThreadsPerBlock;
            int         maxThreadsDim[3];
            int         maxGridSize[3];
            int         clockRate;
            int         multiProcessorCount;
            int         pciBusID;
            int         pciDeviceID;
            size_t      maxSharedMemoryPerMultiProcessor;
            int         gcnArch;
        };

        struct OclAMDGPU : public AMDGPU
        {
            OclAMDGPU() = default;
            OclAMDGPU(oclDeviceProp_t const& prop);

            oclDeviceProp_t properties;
        };

        std::shared_ptr<Hardware> GetCurrentDevice();
        std::shared_ptr<Hardware> GetDevice(int deviceId);
        std::shared_ptr<Hardware> GetDevice(oclDeviceProp_t const& prop);

    } // namespace ocl
} // namespace Tensile

#endif // OCL_HARDWARE_HPP
