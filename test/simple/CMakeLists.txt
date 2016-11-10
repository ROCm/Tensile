cmake_minimum_required(VERSION 2.8.12)

find_package(Cobalt CONFIG)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake )

if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  find_package(OpenCL "1.2" REQUIRED)
  message(STATUS "${Cobalt_BACKEND} backend selected." )
# HIP
elseif( Cobalt_BACKEND MATCHES "HIP")
  find_package( HIP REQUIRED )
  set (CMAKE_CXX_COMPILER ${HIPCC})
  message(STATUS "${Cobalt_BACKEND} backend selected." )
  message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")
endif( )
if(NOT MSVC)
    add_definitions(-std=c++11)
endif()

add_cobalt_lib(SimpleCobaltLib
    SOLUTIONS ${Cobalt_SOLUTIONS}
    BACKEND ${Cobalt_BACKEND}
    
)
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( SimpleCobaltLib PUBLIC opencl )
elseif( Cobalt_BACKEND MATCHES "HIP")
  target_include_directories( SimpleCobaltLib SYSTEM
    PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( SimpleCobaltLib PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( SimpleCobaltLib PUBLIC Cobalt_BACKEND_HIP )
endif()

add_executable(main main.cpp)
target_link_libraries(main SimpleCobaltLib)
