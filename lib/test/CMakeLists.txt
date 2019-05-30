
set(INSTALL_GTEST OFF CACHE BOOL "Install GTest")
add_subdirectory(googletest)

include_directories("." ${GTEST_SOURCE_DIR}/include)

find_package(Boost COMPONENTS filesystem)

if(HIP_FOUND)
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
endif()

add_subdirectory(testlib)

set(test_sources ${test_sources}

    test.cpp

    ContractionProblem_test.cpp
    ContractionSelectionLibrary_test.cpp
    EmbeddedData_test.cpp
    KernelArguments_test.cpp
    TensorDescriptor_test.cpp
    TestData_test.cpp
)

if(USE_LLVM)
    set(test_sources ${test_sources}
        ContractionLibraryLoading_test.cpp
        llvm/LLVMYAMLContraction_test.cpp
        llvm/LibraryPerformance_test.cpp
        )
endif()

add_executable(TensileTests ${test_sources})

if(GTEST_INTEGRATION)
    gtest_discover_tests(TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
else()
    add_test(NAME TensileHostLibraryTests COMMAND TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

target_link_libraries(TensileTests PUBLIC gtest Tensile TensileTestLib Boost::filesystem)

file(COPY sample_library.yaml DESTINATION .)

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")
file(COPY ${SOLUTION_LIBRARY_FILES} DESTINATION "${TEST_DATA_DIR}")

if(HIP_FOUND)
    add_subdirectory(hip)

    target_link_libraries(TensileTests TensileHipTests)
    target_link_libraries(TensileTests "--amdgpu-target=gfx900")
endif()

