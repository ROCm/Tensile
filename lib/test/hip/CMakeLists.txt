
#find_package(ROCM REQUIRED CONFIG PATHS /opt/rocm)

#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

add_library(TensileHipTests STATIC
            HipSolutionAdapter_test.cpp)

message(${gtest_SOURCE_DIR}/include)

target_link_libraries(TensileHipTests hip::host)


target_include_directories(TensileHipTests PRIVATE ${gtest_SOURCE_DIR}/include)

set(tensile_script ../../../../Tensile/Tensile.py) 
set(test_yaml rocblas_sgemm_asm_single_kernel.yaml)
file(COPY ${test_yaml} DESTINATION .)
add_custom_target(test_code_object
                  COMMAND python ${tensile_script} ${test_yaml} code_object
                  DEPENDS ${test_yaml})
#add_dependencies(test_code_object ${test_yaml})



if(false)
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
endif()
