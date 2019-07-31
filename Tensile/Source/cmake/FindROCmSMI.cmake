
find_package(ROCM REQUIRED CONFIG PATHS /opt/rocm)

# For some reason the *_DIR variables have inconsistent values between Tensile and rocBLAS.  Search various paths.
find_path(ROCM_SMI_ROOT "include/rocm_smi/rocm_smi.h"
    PATHS "${ROCM_DIR}/../../.." "${HIP_DIR}/../../../.." "${HIP_DIR}/../../.."
    PATH_SUFFIXES "rocm_smi"
    )

find_library(ROCM_SMI_LIBRARY rocm_smi64
    PATHS "${ROCM_SMI_ROOT}/lib")

add_library(rocm_smi SHARED IMPORTED)

set_target_properties(rocm_smi PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ROCM_SMI_ROOT}/include"
    IMPORTED_LOCATION "${ROCM_SMI_LIBRARY}"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${ROCM_SMI_ROOT}/include")


#set(rocm_smi_root "${hip_LIB_INSTALL_DIR}/../../rocm_smi")
