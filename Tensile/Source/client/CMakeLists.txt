
#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

set(client_sources
    source/BenchmarkTimer.cpp
    source/CSVStackFile.cpp
    source/ClientProblemFactory.cpp
    source/ConvolutionProblem.cpp
    source/DataInitialization.cpp
    source/HardwareMonitor.cpp
    source/HardwareMonitorListener.cpp
    source/MetaRunListener.cpp
    source/PerformanceReporter.cpp
    source/ProgressListener.cpp
    source/Reference.cpp
    source/ReferenceValidator.cpp
    source/ResultFileReporter.cpp
    source/TimingEvents.cpp
    )

find_package(Boost  COMPONENTS program_options filesystem REQUIRED)

add_library(TensileClient STATIC ${client_sources})

find_package(ROCmSMI QUIET)
if(NOT ROCmSMI_FOUND)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${Tensile_DIR}" "${Tensile_DIR}/../Source/cmake" "${CMAKE_HOME_DIRECTORY}/cmake")
    find_package(ROCmSMI REQUIRED)
endif()

target_include_directories(TensileClient PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")# "${rocm_smi_root}/include")

target_link_libraries(TensileClient TensileHost ${Boost_LIBRARIES} rocm_smi)

add_executable(tensile_client main.cpp)

target_link_libraries(tensile_client TensileHost TensileClient ${Boost_LIBRARIES})
if(OpenMP_FOUND)
    target_link_libraries(tensile_client "${OpenMP_EXE_LINKER_FLAGS}")
endif()

