
#find_package( GTest REQUIRED )
set(INSTALL_GTEST OFF)

add_subdirectory(googletest)

set(tensile_test_source
    test.cpp)

set(test_sources
    test.cpp
    GEMMLibrary_test.cpp
    GEMMProblem_test.cpp
    GEMMSolution_test.cpp
    KernelArguments_test.cpp
    TensorDescriptor_test.cpp
)

if(HIP_FOUND)
set(test_sources
    ${test_sources}
	hip/HipSolutionAdapter_test.cpp
)
endif()

add_executable(tests ${test_sources})

target_include_directories (tests PUBLIC "/opt/rocm/rocblas/include" ".")

target_link_libraries(tests gtest Tensile
                      /opt/rocm/rocblas/lib/librocblas.so)
if(HIP_FOUND)
    target_link_libraries(tests ${HIP_LIBRARIES} ${HSA_LIBRARIES})

    set(tensile ../../../Tensile/Tensile.py) 
    set(test_yaml hip/rocblas_sgemm_asm_single_kernel.yaml)
    file(COPY ${test_yaml} DESTINATION hip)
    add_custom_target(code_object
                      COMMAND python ${tensile} ${test_yaml} code_object
                      DEPENDS ${test_yaml}
    )
    #add_dependencies(tests code_object)
endif()



#target_include_directories (test PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
