#include <CL/cl2.hpp>

#include <Tensile/AMDGPU.hpp>

#include <Tensile/ocl/OclHardware.hpp>
#include <Tensile/ocl/OclUtils.hpp>

namespace Tensile
{
    namespace ocl
    {
        OclAMDGPU::OclAMDGPU(oclDeviceProp_t const& prop)
            : AMDGPU(static_cast<AMDGPU::Processor>(prop.gcnArch),
                     prop.multiProcessorCount,
                     std::string(prop.name))
            , properties(prop)
        {
        }

        std::shared_ptr<Hardware> GetCurrentDevice()
        {
            return GetDevice(clGetDevicePropertiesAMD());
        }

        std::shared_ptr<Hardware> GetDevice(int deviceId)
        {
            return GetDevice(clGetDevicePropertiesAMD(deviceId));
        }

        std::shared_ptr<Hardware> GetDevice(int deviceId, cl::Context context)
        {
            return GetDevice(clGetDevicePropertiesAMD(deviceId, context));
        }

        std::shared_ptr<Hardware> GetDevice(oclDeviceProp_t const& prop)
        {
            return std::make_shared<OclAMDGPU>(prop);
        }
    } // namespace hip
} // namespace Tensile
