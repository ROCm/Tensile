
#find_package(ROCM REQUIRED CONFIG PATHS /opt/rocm)

#set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

set(library_yaml TensileKernels.yaml)
file(COPY ${library_yaml} DESTINATION .)

set(lite_yaml KernelsLite.yaml)
file(COPY ${lite_yaml} DESTINATION .)

set(input_yaml sgemm_asm_all_transposes.yaml)
file(COPY ${input_yaml} DESTINATION .)

if(HIP_FOUND)
    set(tensile_script ../../../Tensile/Tensile.py) 

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
        DEPENDS code_object_inputs )

    file(GLOB co_lite_objects
        ${CMAKE_BINARY_DIR}/configs/code_object_lite/4_LibraryClient/build/assembly/*.o
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
