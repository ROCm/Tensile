
#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

set(client_sources
    source/BenchmarkTimer.cpp
    source/CSVStackFile.cpp
    source/ClientProblemFactory.cpp
    source/DataInitialization.cpp
    source/HardwareMonitor.cpp
    source/HardwareMonitorListener.cpp
    source/MetaRunListener.cpp
    source/ProgressListener.cpp
    source/Reference.cpp
    source/ReferenceValidator.cpp
    source/ResultFileReporter.cpp
    source/TimingEvents.cpp
    )

find_package(Boost COMPONENTS program_options)

add_library(TensileClient STATIC ${client_sources})

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${Tensile_DIR}")
find_package(ROCmSMI REQUIRED)

target_include_directories(TensileClient PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")# "${rocm_smi_root}/include")

target_link_libraries(TensileClient TensileHost Boost::program_options rocm_smi)

add_executable(tensile_client main.cpp)

target_link_libraries(tensile_client TensileHost TensileClient Boost::program_options)
if(OpenMP_FOUND)
    target_link_libraries(tensile_client "${OpenMP_EXE_LINKER_FLAGS}")
endif()

