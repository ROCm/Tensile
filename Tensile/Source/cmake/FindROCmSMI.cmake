
if(NOT ROCM_ROOT)
    if(NOT ROCM_DIR)
        set(ROCM_ROOT "/opt/rocm")
    else()
        set(ROCM_DIR "${ROCM_DIR}/../../..")
    endif()
endif()


# For some reason the *_DIR variables have inconsistent values between Tensile and rocBLAS.  Search various paths.
find_path(ROCM_SMI_ROOT "include/rocm_smi/rocm_smi.h"
    PATHS "${ROCM_ROOT}" "${HIP_DIR}/../../../.." "${HIP_DIR}/../../.."
    PATH_SUFFIXES "rocm_smi"
    )
mark_as_advanced(ROCM_SMI_ROOT)

find_library(ROCM_SMI_LIBRARY rocm_smi64
    PATHS "${ROCM_SMI_ROOT}/lib")
mark_as_advanced(ROCM_SMI_LIBRARY)

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( ROCmSMI DEFAULT_MSG ROCM_SMI_LIBRARY ROCM_SMI_ROOT )

add_library(rocm_smi SHARED IMPORTED)

set_target_properties(rocm_smi PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ROCM_SMI_ROOT}/include"
    IMPORTED_LOCATION "${ROCM_SMI_LIBRARY}"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${ROCM_SMI_ROOT}/include")


#set(rocm_smi_root "${hip_LIB_INSTALL_DIR}/../../rocm_smi")
