################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
################################################################################


# static source files
set( CobaltBenchmark_SRC_STATIC
  CobaltBenchmark.cpp
  CobaltBenchmark.h
  ../CobaltLib/include/Cobalt.h
  ../CobaltLib/src/Cobalt.cpp
  ../CobaltLib/src/Solution.h
  ../CobaltLib/src/Solution.cpp
  ../CobaltLib/src/Tools.h
  ../CobaltLib/src/Tools.cpp
)

# generated source files
set( CobaltBenchmark_SRC_GENERATED_STATIC
  ${CobaltBenchmark_DIR_GENERATED}/Kernels/CobaltKernels.h
  ${CobaltBenchmark_DIR_GENERATED}/Solutions/CobaltSolutions.h
  ${CobaltBenchmark_DIR_GENERATED}/Solutions/SolutionTemplateInstantiations.inl
  ${CobaltBenchmark_DIR_GENERATED}/Other/CobaltSolutionCandidates.cpp
  ${CobaltBenchmark_DIR_GENERATED}/Other/CobaltSolutionCandidates.h
)

#include( ${CobaltBenchmark_CMAKE_DYNAMIC} )

############################################################
# CobaltGenBenchmark.py - generates benchmarking files
add_custom_command(
  OUTPUT
    ${CobaltBenchmark_SRC_GENERATED_STATIC}
    ${CobaltBenchmark_SRC_GENERATED_DYNAMIC}
    ${CobaltBenchmark_CMAKE_DYNAMIC}
  COMMAND
  python ${CMAKE_SOURCE_DIR}/CobaltGen/CobaltGenBenchmark.py
      --backend=${Cobalt_BACKEND}
      --input-path=${Cobalt_DIR_PROBLEMS}
      --output-path=${CobaltBenchmark_DIR_GENERATED}
      --optimize-alpha=${Cobalt_OPTIMIZE_ALPHA}
      --optimize-beta=${Cobalt_OPTIMIZE_BETA}
      #echo "python ${CMAKE_SOURCE_DIR}/CobaltGen/CobaltGenBenchmark.py --backend=${Cobalt_BACKEND} --input-path=${Cobalt_DIR_PROBLEMS} --output-path=${Cobalt_DIR_GENERATED} --result-file=${Cobalt_FILE_SOLUTIONS} --optimize-alpha=${Cobalt_OPTIMIZE_ALPHA} --optimize-beta=${Cobalt_OPTIMIZE_BETA}"
  DEPENDS ${CobaltGen_SRC} ${Cobalt_XML_PROBLEMS_FILE}
)

# benchmark executable
add_executable( ${CobaltBenchmark_NAME} EXCLUDE_FROM_ALL
  ${CobaltBenchmark_SRC_STATIC}
  ${CobaltBenchmark_SRC_GENERATED_STATIC}
  ${CobaltBenchmark_SRC_GENERATED_DYNAMIC}
  ${CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC}
  ${CobaltBenchmark_SolutionFiles_GENERATED_DYNAMIC}
  ${CobaltGen_SRC}
)

#set_property(TARGET ${CobaltBenchmark_NAME} PROPERTY CXX_STANDARD 11 )
source_group( CobaltGen\\Scripts FILES ${CobaltGen_SRC} )
source_group( CobaltGen\\Kernels FILES ${CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC} )
source_group( CobaltGen\\Solutions FILES ${CobaltBenchmark_SolutionFiles_GENERATED_DYNAMIC} )
source_group( CobaltGen\\Benchmark FILES
  ${CobaltBenchmark_SRC_GENERATED_STATIC}
  ${CobaltBenchmark_SRC_GENERATED_DYNAMIC}
)

# benchmark links Cobalt
target_link_libraries( ${CobaltBenchmark_NAME} PRIVATE ${CobaltLogger_NAME} )

# include Cobalt headers
target_include_directories( ${CobaltBenchmark_NAME}
  PUBLIC  $<BUILD_INTERFACE:${CobaltLib_DIRS_INCLUDE}>
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

if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  target_compile_definitions( ${CobaltBenchmark_NAME} PRIVATE -DCobalt_DIR_SOLUTIONS="${Cobalt_DIR_SOLUTIONS}" )
elseif( Cobalt_BACKEND MATCHES "HIP")
  add_definitions(
    -DCobalt_DIR_SOLUTIONS=\\\"${Cobalt_DIR_SOLUTIONS}\\\")
endif()
