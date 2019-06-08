
#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

set(client_sources
    BenchmarkTimer.cpp
    ClientProblemFactory.cpp
    DataInitialization.cpp
    HardwareMonitor.cpp
    MetaRunListener.cpp
    Reference.cpp
    ReferenceValidator.cpp
    TimingEvents.cpp
    )

find_package(Boost COMPONENTS program_options)

add_library(TensileClient STATIC ${client_sources})

set(rocm_smi_root "${hip_LIB_INSTALL_DIR}/../rocm_smi")

target_include_directories(TensileClient PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${rocm_smi_root}/include")

target_link_libraries(TensileClient Tensile Boost::program_options "${rocm_smi_root}/lib/librocm_smi64.so")

add_executable(tensile_client main.cpp)

target_link_libraries(tensile_client Tensile TensileClient Boost::program_options)
target_link_libraries(tensile_client "--amdgpu-target=gfx900")


