
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

set(client_sources
    Reference.cpp
    DataInitialization.cpp
    ClientProblemFactory.cpp)

find_package(Boost COMPONENTS program_options)

add_library(TensileClient STATIC ${client_sources})

target_include_directories(TensileClient PUBLIC .)

target_link_libraries(TensileClient Tensile Boost::program_options)

add_executable(tensile_client main.cpp)

target_link_libraries(tensile_client Tensile TensileClient Boost::program_options)
target_link_libraries(tensile_client "--amdgpu-target=gfx900")


