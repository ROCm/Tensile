
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
  ${CobaltBenchmark_DIR_GENERATED}/Other/CobaltSolutionCandidates.cpp
  ${CobaltBenchmark_DIR_GENERATED}/Other/CobaltSolutionCandidates.h
  ${CobaltBenchmark_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl
)


if( NOT EXISTS ${CobaltBenchmark_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl)
  MESSAGE( "Creating dummy SolutionTemplateInstantiations.inl for benchmark")
  file( WRITE ${CobaltBenchmark_DIR_GENERATED}/Other/SolutionTemplateInstantiations.inl
      ""
  )
endif()


set( CobaltBenchmark_CMAKE_DYNAMIC
  ${CobaltBenchmark_DIR_GENERATED}/Other/CobaltBenchmark.cmake
)
if( NOT EXISTS ${CobaltBenchmark_CMAKE_DYNAMIC})
  file( WRITE ${CobaltBenchmark_CMAKE_DYNAMIC}
      "include( \${CobaltBenchmark_KernelFiles_CMAKE_DYNAMIC} )\n"
      "include( \${CobaltBenchmark_SolutionFiles_CMAKE_DYNAMIC} )\n"
      "set( CobaltBenchmark_SRC_GENERATED_DYNAMIC\n"
      "  \${CobaltBenchmark_KernelFiles_GENERATED_DYNAMIC}\n"
      "  \${CobaltBenchmark_SolutionFiles_GENERATED_DYNAMIC} )\n"
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
      --input-path=${CobaltBenchmark_DIR_PROBLEMS}
      --output-path=${CobaltBenchmark_DIR_GENERATED}
      --optimize-alpha=${Cobalt_OPTIMIZE_ALPHA}
      --optimize-beta=${Cobalt_OPTIMIZE_BETA}
      #echo "python ${CMAKE_SOURCE_DIR}/CobaltGen/CobaltGenBenchmark.py --backend=${Cobalt_BACKEND} --input-path=${Cobalt_DIR_PROBLEMS} --output-path=${Cobalt_DIR_GENERATED} --result-file=${Cobalt_FILE_SOLUTIONS} --optimize-alpha=${Cobalt_OPTIMIZE_ALPHA} --optimize-beta=${Cobalt_OPTIMIZE_BETA}"
  DEPENDS ${CobaltGen_SRC}
)


# benchmark executable
add_executable( ${CobaltBenchmark_NAME}
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

# gemm links Cobalt
target_link_libraries( ${CobaltBenchmark_NAME}
  ${CobaltLib_NAME}
)

# include Cobalt headers
include_directories(
  ${CobaltLib_DIRS_INCLUDE}
  ${CobaltBenchmark_DIR_GENERATED}
  ${CobaltBenchmark_DIR_GENERATED}/Kernels
  ${CobaltBenchmark_DIR_GENERATED}/Solutions
  ${CobaltBenchmark_DIR_GENERATED}/Other
)

# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  include_directories( ${OPENCL_INCLUDE_DIRS} )
  target_link_libraries( ${CobaltBenchmark_NAME} ${OPENCL_LIBRARIES} )
  MESSAGE( "CobaltBenchmark linking ${OPENCL_LIBRARIES}")
elseif( Cobalt_BACKEND MATCHES "HIP")
  MESSAGE( "CobaltBenchmark linking HIP: ${HIP_INCLUDE_DIRS}" )
  include_directories( ${HIP_INCLUDE_DIRS} )
  # target_link_libraries( ${CobaltBenchmark_NAME} ${HSA_LIBRARIES} )
endif()

############################################################
# CobaltBenchmark - generates ProblemSolutionMap.xml
#add_custom_command(
#  OUTPUT ${Cobalt_FILE_SOLUTIONS}
#  COMMAND ${CobaltBenchmark_NAME}
#  DEPENDS
#    ${CobaltBenchmark_SRC_STATIC}
#    ${CobaltBenchmark_SRC_GENERATED_STATIC}
#    ${CobaltBenchmark_SRC_GENERATED_DYNAMIC}
#)

add_definitions( -DCobalt_LOGGER_ENABLED=1)
add_definitions( -DCobalt_SOLVER_ENABLED=0)



add_definitions(
  -DCobaltBenchmark_DIR_SOLUTIONS=\\\"${CobaltBenchmark_DIR_SOLUTIONS}\\\")
#-DCobaltBenchmark_DIR_SOLUTIONS="${CobaltBenchmark_DIR_SOLUTIONS}")
