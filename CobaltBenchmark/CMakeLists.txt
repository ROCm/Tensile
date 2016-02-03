set( CobaltBenchmark_NAME CobaltBenchmark )

############################################################
# defined externally
#set( Cobalt_DIR_GENERATED ${CMAKE_BINARY_DIR}/CobaltGenerated ) # temp
#set( Cobalt_DIR_PROBLEMS ${Cobalt_DIR_GENERATED}/AppProblemLogs ) # input
#set( Cobalt_PSM ${Cobalt_PSM}/Cobal_PSM_PREFIX.xml ) # output


# static source files
set( CobaltBenchmark_SRC_STATIC
  CobaltBenchmark.cpp
  ../CobaltLib/src/Tools.h
  ../CobaltLib/src/Tools.cpp
)

# generated source files
set( CobaltBenchmark_SRC_GENERATED_STATIC
  ${Cobalt_DIR_GENERATED}/Kernels/CobaltKernels.h
  ${Cobalt_DIR_GENERATED}/Solutions/CobaltSolutions.h
  ${Cobalt_DIR_GENERATED}/Benchmark/CobaltSolutionCandidates.cpp
  ${Cobalt_DIR_GENERATED}/Benchmark/CobaltSolutionCandidates.h
)

#set( CobaltBenchmark_SRC_GENERATED_DYNAMIC
  # from CobaltBenchmark.cmake
#)
set( CobaltBenchmark_CMAKE_DYNAMIC
  ${Cobalt_DIR_GENERATED}/Benchmark/CobaltBenchmark.cmake
)
if( NOT EXISTS ${CobaltBenchmark_CMAKE_DYNAMIC})
  file( WRITE ${CobaltBenchmark_CMAKE_DYNAMIC}
      "include( \${Cobalt_KernelFiles_CMAKE_DYNAMIC} )\n"
      "include( \${Cobalt_SolutionFiles_CMAKE_DYNAMIC} )\n"
      "set( CobaltBenchmark_SRC_GENERATED_DYNAMIC\n"
      "  \${Cobalt_KernelFiles_GENERATED_DYNAMIC}\n"
      "  \${Cobalt_SolutionFiles_GENERATED_DYNAMIC} )\n"
  )
endif()
include( ${CobaltBenchmark_CMAKE_DYNAMIC} )

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
    --output-path=${Cobalt_DIR_GENERATED}
    --result-file=${Cobalt_FILE_PSM}
  DEPENDS ${CobaltGenBenchmark}
)

source_group( source FILES
  ${CobaltBenchmark_SRC_STATIC}
  ${CobaltBenchmark_SRC_GENERATED_STATIC}
  ${CobaltBenchmark_SRC_GENERATED_DYNAMIC}
)
source_group( CobaltGenBenchmark FILES ${CobaltGenBenchmark_SRC} )

# benchmark executable
add_executable( ${CobaltBenchmark_NAME}
  ${CobaltBenchmark_SRC_STATIC}
  ${CobaltBenchmark_SRC_GENERATED_STATIC}
  ${CobaltBenchmark_SRC_GENERATED_DYNAMIC}
)

# gemm links Cobalt
target_link_libraries( ${CobaltBenchmark_NAME}
  ${CobaltLib_NAME}
)

# include Cobalt headers
include_directories(
  ${CobaltLib_DIRS_INCLUDE}
  ${Cobalt_DIR_GENERATED}/Benchmark
  ${Cobalt_DIR_GENERATED}/Kernels
  ${Cobalt_DIR_GENERATED}/Solutions
)

# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  include_directories( ${OpenCL_INCLUDE_DIR} )
  target_link_libraries( ${CobaltBenchmark_NAME} ${OpenCL_LIBRARY} )
endif()

############################################################
# CobaltBenchmark - generates ProblemSolutionMap.xml
add_custom_command(
  OUTPUT ${Cobalt_FILE_PSM}
  COMMAND ${CobaltBenchmark_NAME}
  DEPENDS
    ${CobaltBenchmark_SRC_STATIC}
    ${CobaltBenchmark_SRC_GENERATED_STATIC}
    ${CobaltBenchmark_SRC_GENERATED_DYNAMIC}
)

