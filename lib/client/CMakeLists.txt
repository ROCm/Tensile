
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

set(client_sources
    BenchmarkTimer.cpp
    ClientProblemFactory.cpp
    DataInitialization.cpp
    MetaRunListener.cpp
    Reference.cpp
    ReferenceValidator.cpp
    TimingEvents.cpp
    )

find_package(Boost COMPONENTS program_options)

add_library(TensileClient STATIC ${client_sources})

target_include_directories(TensileClient PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(TensileClient Tensile Boost::program_options)

add_executable(tensile_client main.cpp)

target_link_libraries(tensile_client Tensile TensileClient Boost::program_options)
target_link_libraries(tensile_client "--amdgpu-target=gfx900")


