#pragma once

#include <Tensile/Tensile.hpp>
#include <Tensile/ocl/OclFwd.hpp>

#include <mutex>

namespace Tensile
{
    namespace ocl
    {
        class SolutionAdapter final : public Tensile::SolutionAdapter
        {
        public:
            SolutionAdapter();
            SolutionAdapter(bool debug);
            SolutionAdapter(bool debug, std::string const& name);
            SolutionAdapter(bool               debug,
                            std::string const& name,
                            cl::Context        context,
                            cl::Device         device);
            ~SolutionAdapter() final = default;

            std::string name() const final;

            void loadCodeObjectFile(std::string const& path);
            void loadCodeObjectBytes(std::vector<uint8_t> const& bytes);

            void loadEmbeddedCodeObjects();
            void loadEmbeddedCodeObjects(std::string const& key);

            void launchKernel(KernelInvocation const& kernel);
            void launchKernel(KernelInvocation const& kernel,
                              cl::CommandQueue        stream,
                              cl::Event*              timingEvent = nullptr);

            void launchKernels(std::vector<KernelInvocation> const& kernels);
            void launchKernels(std::vector<KernelInvocation> const& kernels,
                               cl::CommandQueue                     stream,
                               cl::Event*                           timingEvent = nullptr);

            void launchKernels(std::vector<KernelInvocation> const& kernels,
                               cl::CommandQueue                     stream,
                               std::vector<cl::Event>&              timingEvents);

            void initKernel(std::string const& name);

        private:
            cl::Kernel getKernel(std::string const& name);
            void       addModule(std::string const& name, cl::Program const& module);
            void addModules(std::string const& groupName, std::vector<cl::Program> const& modules);

            std::mutex m_access;

            std::vector<cl::Program>                    m_modules;
            cl::Context                                 m_context;
            cl::Device                                  m_device;
            std::unordered_map<std::string, cl::Kernel> m_kernels;
            bool                                        m_debug           = false;
            bool                                        m_debugSkipLaunch = false;
            std::string                                 m_name            = "OclSolutionAdapter";

            std::vector<std::string> m_loadedModuleNames;

            friend std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter);
        };

        std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter);

    } // namespace ocl
} // namespace Tensile
