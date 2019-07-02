
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

#add_definitions(-D__HIP_PLATFORM_HCC__)
#set(HIP_INCLUDE_DIRS /opt/rocm/hip/include)
#set(HIP_LIBRARIES /opt/rocm/hip/lib/libhip_hcc.so)

add_library(TensileHip STATIC 
            HipSolutionAdapter.cpp
            HipHardware.cpp)

message( TensileHip SYSTEM ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS} )

include_directories(${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS})

