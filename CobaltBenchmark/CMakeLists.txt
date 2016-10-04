
set( CobaltBenchmark_NAME CobaltBenchmark )

get_filename_component(Cobalt_DIR ${CMAKE_SOURCE_DIR} PATH)

set( CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${Cobalt_DIR}/cmake )

############################################################
# require C++11
if(MSVC)
  # object-level build parallelism for VS, not just target-level
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP" )
  set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else()
endif()
add_definitions( "-std=c++11" )

# OpenCL 1.2
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  find_package(OpenCL "1.2" REQUIRED)
  message(STATUS "${Cobalt_BACKEND} backend selected." )
# HIP
elseif( Cobalt_BACKEND MATCHES "HIP")
  find_package( HIP REQUIRED )
  message(STATUS "${Cobalt_BACKEND} backend selected." )
  message(STATUS "HIP_PLATFORM = ${HIP_PLATFORM}")
endif( )


# source files
file(GLOB CobaltBenchmark_SRC
    ${CobaltBenchmark_DIR_GENERATED}/Kernels/*.cpp
    ${CobaltBenchmark_DIR_GENERATED}/Solutions/*.cpp
    ${CobaltBenchmark_DIR_GENERATED}/Other/*.cpp
    ${Cobalt_DIR}/CobaltLib/src/*.cpp
)
# benchmark executable
add_executable( ${CobaltBenchmark_NAME}
    CobaltBenchmark.cpp
    ${CobaltBenchmark_SRC}
)

##################################################
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( ${CobaltBenchmark_NAME} PRIVATE opencl )
  target_compile_definitions( ${CobaltBenchmark_NAME} PUBLIC -DCobalt_BACKEND_OPENCL12=1 -DCobalt_BACKEND_HIP=0 )
elseif( Cobalt_BACKEND MATCHES "HIP")
  target_include_directories( ${CobaltBenchmark_NAME}
    PUBLIC  $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )
#  target_link_libraries( ${CobaltBenchmark_NAME} PRIVATE ${HSA_LIBRARIES} )
  target_link_libraries( ${CobaltBenchmark_NAME} PUBLIC $<BUILD_INTERFACE:${HSA_LIBRARIES}> )
  target_compile_definitions( ${CobaltBenchmark_NAME} PUBLIC -DCobalt_BACKEND_OPENCL12=0 -DCobalt_BACKEND_HIP=1 )
endif()

# include Cobalt headers
target_include_directories( ${CobaltBenchmark_NAME}
  PUBLIC  $<BUILD_INTERFACE:${Cobalt_DIR}/CobaltLib/include>
          $<BUILD_INTERFACE:${Cobalt_DIR}/CobaltLib/src>
          $<BUILD_INTERFACE:${CobaltBenchmark_DIR_GENERATED}>
          $<BUILD_INTERFACE:${CobaltBenchmark_DIR_GENERATED}/Kernels>
          $<BUILD_INTERFACE:${CobaltBenchmark_DIR_GENERATED}/Solutions>
          $<BUILD_INTERFACE:${CobaltBenchmark_DIR_GENERATED}/Other>
)

# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_link_libraries( ${CobaltBenchmark_NAME} PRIVATE opencl )
elseif( Cobalt_BACKEND MATCHES "HIP")
  message( STATUS "CobaltBenchmark linking HIP: ${HIP_INCLUDE_DIRS}" )
  include_directories( ${HIP_INCLUDE_DIRS} )
endif()

############################################################
# CobaltBenchmark - generates ProblemSolutionMap.xml

target_compile_definitions( ${CobaltBenchmark_NAME} PRIVATE -DCobalt_SOLVER_ENABLED=0 -DCobalt_LOGGER_ENABLED=1 )

target_compile_definitions( ${CobaltBenchmark_NAME} PRIVATE -DCobalt_DIR_SOLUTIONS="${Cobalt_DIR_SOLUTIONS}" )
