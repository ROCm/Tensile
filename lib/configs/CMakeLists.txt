
#find_package(ROCM REQUIRED CONFIG PATHS /opt/rocm)

#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

set(library_yaml TensileKernels.yaml)
file(COPY ${library_yaml} DESTINATION .)

set(lite_yaml KernelsLite.yaml)
file(COPY ${lite_yaml} DESTINATION .)

set(input_yaml sgemm_asm_all_transposes.yaml)
file(COPY ${input_yaml} DESTINATION .)

set(lite_configs lite_configs)
file(COPY ${lite_configs} DESTINATION .)

set (asm_only_yaml  ${CMAKE_CURRENT_SOURCE_DIR}/../../Tensile/Configs/rocblas_sgemm_asm_only.yaml)

set(tensile_library_options --merge-files)
if(CMAKE_BUILD_TYPE EQUAL "Debug")
    set(tensile_library_options ${tensile_library_options} --library-print-debug)
endif()

if(HIP_FOUND)
    set(tensile_script ${CMAKE_CURRENT_SOURCE_DIR}/../../Tensile/Tensile.py) 
    set(tensile_library_script ${CMAKE_CURRENT_SOURCE_DIR}/../../Tensile/TensileCreateLibrary.py) 

    add_custom_target(code_object_inputs
                      COMMAND python ${tensile_script} ${input_yaml} code_object
                      DEPENDS ${input_yaml})

    set(co_transposes Cijk_Ailk_Bjlk
                      Cijk_Ailk_Bljk
                      Cijk_Alik_Bjlk
                      Cijk_Alik_Bljk)

    foreach(_trans ${co_transposes})
        set(co_object_files ${co_object_files}
            code_object/1_BenchmarkProblems/${_trans}_SB_00/00_Final/source/assembly/${_trans}_SB_MT128x128x08_K1.o)
    endforeach()

    add_custom_target(code_object
        COMMAND ${HIP_BIN_INSTALL_DIR}/hcc -target amdgcn--amdhsa ${co_object_files} -o TensileKernels.co
        DEPENDS code_object_inputs)

    file(GLOB co_lite_objects
        ${CMAKE_BINARY_DIR}/configs/lite_library/assembly/*.o
        )

    add_custom_target(code_object_lite_bench
        COMMAND python ${tensile_script}  ${asm_only_yaml} code_object_lite
        )

    add_custom_target(code_object_lite_library
        WORKING_DIRECTORY lite_library
        COMMAND python ${tensile_library_script} ../${lite_configs} Tensile HIP ${tensile_library_options}
        )

    add_custom_target(code_object_lite
        COMMAND ${HIP_BIN_INSTALL_DIR}/hcc -target amdgcn--amdhsa ${co_lite_objects} -o KernelsLite.co)
endif()

if(false)
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
endif()
