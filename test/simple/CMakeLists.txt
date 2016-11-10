cmake_minimum_required(VERSION 2.8.12)

find_package(Tensile CONFIG)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake )

if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  find_package(OpenCL "1.2" REQUIRED)
  message(STATUS "${Tensile_BACKEND} backend selected." )
# HIP
elseif( Tensile_BACKEND MATCHES "HIP")
  find_package( HIP REQUIRED )
  set (CMAKE_CXX_COMPILER ${HIPCC})
  message(STATUS "${Tensile_BACKEND} backend selected." )
  message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")
endif( )
if(NOT MSVC)
    add_definitions(-std=c++11)
endif()

add_tensile_lib(SimpleTensileLib
    SOLUTIONS ${Tensile_SOLUTIONS}
    BACKEND ${Tensile_BACKEND}
    
)
if( Tensile_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( SimpleTensileLib PUBLIC opencl )
elseif( Tensile_BACKEND MATCHES "HIP")
  target_include_directories( SimpleTensileLib SYSTEM
    PUBLIC $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
  target_link_libraries( SimpleTensileLib PUBLIC ${HSA_LIBRARIES} )
  target_compile_definitions( SimpleTensileLib PUBLIC Tensile_BACKEND_HIP )
endif()

add_executable(main main.cpp)
target_link_libraries(main SimpleTensileLib)
