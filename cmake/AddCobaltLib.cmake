include(CMakeParseArguments)

function(add_cobalt_lib NAME)
    set(options OPTIMIZE_ALPHA OPTIMIZE_BETA ENABLE_LOGGER EXCLUDE_FROM_ALL)
    set(oneValueArgs SOLUTIONS PROBLEMS BACKEND)
    set(multiValueArgs)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(CobaltLib_DIR_GENERATED "${PROJECT_BINARY_DIR}/cobalt/${NAME}")

    if(PARSE_SOLUTIONS)

        set(CobaltLib_ENABLE_SOLVER 1)
        execute_process(
            COMMAND
            ${PYTHON_EXECUTABLE} ${Cobalt_DIR}/CobaltGen/CobaltGenBackend.py
            --backend=${PARSE_BACKEND}
            --input-path=${PARSE_SOLUTIONS}
            --output-path=${CobaltLib_DIR_GENERATED}
            --optimize-alpha=${PARSE_OPTIMIZE_ALPHA}
            --optimize-beta=${PARSE_OPTIMIZE_BETA}
        )
        # Glob CobaltLib source files
        file(GLOB CobaltLib_SRC
            ${Cobalt_DIR}/CobaltLib/src/*.cpp
            ${CobaltLib_DIR_GENERATED}/Kernels/*.cpp
            ${CobaltLib_DIR_GENERATED}/Solutions/*.cpp
            ${CobaltLib_DIR_GENERATED}/Other/*.cpp
        )

    else()
        file( WRITE ${CobaltLib_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl "")
        # Glob CobaltLib source files
        file(GLOB CobaltLib_SRC
            ${Cobalt_DIR}/CobaltLib/src/*.cpp
        )
        set(CobaltLib_ENABLE_SOLVER 0)

    endif()

    set(options)
    if(PARSE_EXCLUDE_FROM_ALL)
        list(APPEND options EXCLUDE_FROM_ALL)
    endif()
    add_library(${NAME} ${options} ${CobaltLib_SRC})

    target_include_directories(${NAME}
        PUBLIC  $<BUILD_INTERFACE:${Cobalt_DIR}/CobaltLib/include>
                $<BUILD_INTERFACE:${Cobalt_DIR}/CobaltLib/src>
                $<BUILD_INTERFACE:${CobaltLib_DIR_GENERATED}>
                $<BUILD_INTERFACE:${CobaltLib_DIR_GENERATED}/Kernels>
                $<BUILD_INTERFACE:${CobaltLib_DIR_GENERATED}/Solutions>
                $<BUILD_INTERFACE:${CobaltLib_DIR_GENERATED}/Other>
                $<INSTALL_INTERFACE:include>
    )

    if( PARSE_BACKEND MATCHES "OpenCL_1.2")
        target_compile_definitions( ${NAME} PUBLIC -DCobalt_BACKEND_OPENCL12=1 -DCobalt_BACKEND_HIP=0 )
    elseif( PARSE_BACKEND MATCHES "HIP")
        target_compile_definitions( ${NAME} PUBLIC -DCobalt_BACKEND_OPENCL12=0 -DCobalt_BACKEND_HIP=1 )
    endif()

    if( ${PARSE_ENABLE_LOGGER} )
        set(CobaltLib_ENABLE_LOGGER 1)
    else()
        set(CobaltLib_ENABLE_LOGGER 0)
    endif()

    target_compile_definitions( ${NAME} PRIVATE 
        -DCobalt_SOLVER_ENABLED=${CobaltLib_ENABLE_SOLVER} 
        -DCobalt_LOGGER_ENABLED=${CobaltLib_ENABLE_LOGGER}
    )

    if(PARSE_PROBLEMS)
        if (CMAKE_CXX_COMPILER MATCHES ".*hipcc")
            # hipcc is a pearl script, so it requires a lot of extra escaping
            target_compile_definitions(${NAME} PUBLIC -DCobalt_DIR_PROBLEMS=\\\"${PARSE_PROBLEMS}\\\")
        else()
            target_compile_definitions(${NAME} PUBLIC -DCobalt_DIR_PROBLEMS="${PARSE_PROBLEMS}")
        endif()
    endif()

endfunction()
