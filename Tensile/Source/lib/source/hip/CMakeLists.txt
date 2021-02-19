
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

add_library(TensileHip STATIC
            HipSolutionAdapter.cpp
            HipHardware.cpp)

message( TensileHip SYSTEM ${HIP_INCLUDE_DIRS} )

include_directories(${HIP_INCLUDE_DIRS})

