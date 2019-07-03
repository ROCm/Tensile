
if(NOT TENSILE_DISABLE_TESTS)
    enable_testing()

    if(CMAKE_VERSION VERSION_GREATER 3.9)
        include(GoogleTest)
        option(GTEST_INTEGRATION "Enable Googletest integration with CMake." ON)
    else()
        option(GTEST_INTEGRATION "Enable Googletest integration with CMake." OFF)
    endif()
endif()

set(INSTALL_GTEST OFF CACHE BOOL "Install GTest")
add_subdirectory(googletest)

include_directories("." ${GTEST_SOURCE_DIR}/include)

find_package(Boost COMPONENTS filesystem)

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")
file(COPY ${SOLUTION_LIBRARY_FILES} DESTINATION "${TEST_DATA_DIR}")

add_subdirectory(testlib)

set(test_sources ${test_sources}

    test.cpp

    ContractionProblem_test.cpp
    ContractionSelectionLibrary_test.cpp
    EmbeddedData_test.cpp
    KernelArguments_test.cpp
    ProjectedPerformance_test.cpp
    TensorDescriptor_test.cpp
    TestData_test.cpp
    Utils_test.cpp
)

if(USE_LLVM)
    set(test_sources ${test_sources}
        ContractionLibraryLoading_test.cpp
        llvm/LLVMYAMLContraction_test.cpp
        llvm/LibraryPerformance_test.cpp
        )
endif()

if(HIP_FOUND)
    add_subdirectory(hip)
endif()

message(STATUS "Test sources: ${test_sources}")
add_executable(TensileTests ${test_sources})

if(GTEST_INTEGRATION)
    gtest_discover_tests(TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
else()
    add_test(NAME TensileHostLibraryTests COMMAND TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

target_link_libraries(TensileTests gtest Tensile TensileTestLib Boost::filesystem)

if(HIP_FOUND)

    target_link_libraries(TensileTests ${HIP_TEST_LIBRARIES})
    target_link_libraries(TensileTests "--amdgpu-target=gfx900")
endif()

