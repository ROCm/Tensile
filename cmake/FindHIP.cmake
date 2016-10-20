################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################


find_package(CUDA)

if( CUDA_INCLUDE_DIRS AND CUDA_VERSION AND CUDA_NVCC_EXECUTABLE)
    message(STATUS "CUDA_VERSION = ${CUDA_VERSION}")
    message(STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_NVCC_EXECUTABLE = ${CUDA_NVCC_EXECUTABLE}")

    set( HIP_PLATFORM "nvcc" )

    #export the environment variable, so that HIPCC can find it.
    set(ENV{HIP_PLATFORM} nvcc)

else()
    find_package(HCC)

    message(STATUS "HCC_FOUND = ${HCC_FOUND}")
    message(STATUS "HCC = ${HCC}")
    message(STATUS "HCC_INCLUDE_DIRS = ${HCC_INCLUDE_DIRS}")
    message(STATUS "HSA_LIBRARIES = ${HSA_LIBRARIES}")

    if( ${HCC_FOUND} STREQUAL  "TRUE" )

    if (NOT DEFINED HIP_PATH)
      find_path( HIP_INCLUDE_DIR
            NAMES
                hip_runtime.h
            PATHS
                ENV HIP_PATH
                /opt/rocm
            PATH_SUFFIXES
                /include/hip
                /include
        )

      set (HIP_INCLUDE_DIRS ${HIP_INCLUDE_DIR})
      set (HIP_PATH ${HIP_INCLUDE_DIR})
      if (NOT DEFINED ENV{HIP_PATH})
              set( ENV{HIP_PATH} ${HIP_PATH})
      endif( )
    endif()

        message(STATUS "ENV HIP_PATH = $ENV{HIP_PATH}")

        find_program(HIPCC
            NAMES  hipcc
            PATHS
                ENV HIP_PATH
                /opt/rocm
            PATH_SUFFIXES
                /bin
            )

            message(STATUS "HIPCC = ${HIPCC}")

        if( DEFINED HIPCC)

            set( HIP_PLATFORM "hcc" )
            #export the environment variable, so that HIPCC can find it.
            set(ENV{HIP_PLATFORM} "hcc")
            # set (CMAKE_CXX_COMPILER ${HIPCC})

        else()
            message(SEND_ERROR "Did not find HIPCC")
        endif()
    else()
        message(SEND_ERROR "hcc not found")
    endif()

endif()
