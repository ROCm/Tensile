
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

add_library(TensileOcl STATIC
            OclHardware.cpp
            OclSolutionAdapter.cpp
            OclUtils.cpp)

message( TensileOcl SYSTEM ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS} )

include_directories(${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS})

