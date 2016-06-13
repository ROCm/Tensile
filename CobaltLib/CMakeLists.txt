################################################################################
# CobaltLogger & CobaltLib
################################################################################

############################################################
# CobaltLib source files
set( CobaltLib_SRC_STATIC
  src/Cobalt.cpp
  src/DeviceProfile.cpp
  src/DeviceProfile.h
  src/Logger.cpp
  src/Logger.h
  src/MathTemplates.cpp
  src/MathTemplates.h
  src/Problem.cpp
  src/Problem.h
  src/SolutionTensorContractionCPU.cpp
  src/SolutionTensorContractionCPU.h
  src/Solution.cpp
  src/Solution.h
  src/StructOperations.cpp
  src/StructOperations.h
  src/Tensor.cpp
  src/Tensor.h
  src/Tools.cpp
  src/Tools.h
)

# TODO does this need to change to Windows vs Linux?
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  add_definitions(-DCobaltLib_DIR_PROBLEMS="${CobaltLib_DIR_PROBLEMS}")
  add_definitions(-DCobaltLib_DIR_SOLUTIONS="${CobaltLib_DIR_SOLUTIONS}")
  add_definitions(-DCobaltLib_DIR_GENERATED="${CobaltLib_DIR_GENERATED}")
elseif( Cobalt_BACKEND MATCHES "HIP")
  add_definitions(-DCobaltLib_DIR_PROBLEMS=\\\"${CobaltLib_DIR_PROBLEMS}\\\")
  add_definitions(-DCobaltLib_DIR_SOLUTIONS=\\\"${CobaltLib_DIR_SOLUTIONS}\\\")
  add_definitions(-DCobaltLib_DIR_GENERATED=\\\"${CobaltLib_DIR_GENERATED}\\\")
endif()

################################################################################
# CobaltLogger
################################################################################


############################################################
# CobaltLogger
add_library( ${CobaltLogger_NAME}
  ${CobaltLib_API}
  ${CobaltLib_SRC_STATIC}
  #${CobaltLib_SRC_GENERATED_STATIC}
  #${CobaltLib_SRC_GENERATED_DYNAMIC}
  #${CobaltGen_SRC}
  )
set_target_properties( ${CobaltLogger_NAME} PROPERTIES COMPILE_FLAGS
  "-DCobalt_SOLVER_ENABLED=0 -DCobalt_LOGGER_ENABLED=1 -fPIC" )



################################################################################
# CobaltLib
################################################################################

# generated src files
set( CobaltLib_SRC_GENERATED_STATIC
  ${CobaltLib_DIR_GENERATED}/Kernels/CobaltKernels.h
  ${CobaltLib_DIR_GENERATED}/Solutions/CobaltSolutions.h
  ${CobaltLib_DIR_GENERATED}/Other/CobaltGetSolution.cpp
  ${CobaltLib_DIR_GENERATED}/Other/CobaltGetSolution.h
)

#include( ${CobaltLib_CMAKE_DYNAMIC} )


################################################################################
# CobaltLib log-only mode
# - generate static and dynamic any files


############################################################
# CobaltGenBackend.py - generates full library backend files
add_custom_command(
  OUTPUT
    ${CobaltLib_SRC_GENERATED_STATIC}
    ${CobaltLib_SRC_GENERATED_DYNAMIC}
    ${CobaltLib_CMAKE_DYNAMIC}
  COMMAND
    python ${CMAKE_SOURCE_DIR}/CobaltGen/CobaltGenBackend.py
    --backend=${Cobalt_BACKEND}
    --input-path=${Cobalt_DIR_SOLUTIONS}
    --output-path=${CobaltLib_DIR_GENERATED}
  DEPENDS ${CobaltGen_SRC} ${Cobalt_XML_SOLUTIONS_FILE}
)




############################################################
# Cobalt include directories
include_directories(
  src
  ${CobaltLib_DIRS_INCLUDE}
  ${CobaltLib_DIR_GENERATED}
  ${CobaltLib_DIR_GENERATED}/Kernels
  ${CobaltLib_DIR_GENERATED}/Solutions
  ${CobaltLib_DIR_GENERATED}/Other
  )

############################################################
# CobaltLib
add_library( ${CobaltLib_NAME}
  ${CobaltLib_API}
  ${CobaltLib_SRC_STATIC}
  ${CobaltLib_SRC_GENERATED_STATIC}
  ${CobaltLib_SRC_GENERATED_DYNAMIC}
  ${CobaltGen_SRC}
  )
#set_property(TARGET ${CobaltLib_NAME} PROPERTY CXX_STANDARD 11 )
if( ${Cobalt_ENABLE_LOGGER} )
  set_target_properties( ${CobaltLib_NAME} PROPERTIES COMPILE_FLAGS
    "-DCobalt_SOLVER_ENABLED=1 -DCobalt_LOGGER_ENABLED=1 -fPIC" )
else()
  set_target_properties( ${CobaltLib_NAME} PROPERTIES COMPILE_FLAGS
    "-DCobalt_SOLVER_ENABLED=0 -DCobalt_LOGGER_ENABLED=1 -fPIC" )
endif()

source_group(CobaltGen\\Scripts FILES ${CobaltGen_SRC})
source_group(CobaltGen\\Generated FILES
  ${CobaltLib_SRC_GENERATED_STATIC}
  ${CobaltLib_SRC_GENERATED_DYNAMIC}
)

############################################################
# Cobalt depends on OpenCL
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  message( STATUS "CobaltLib linking OpenCL: ${OPENCL_INCLUDE_DIRS}" )
  include_directories( ${OPENCL_INCLUDE_DIRS} )
  target_link_libraries( ${CobaltLib_NAME} ${OPENCL_LIBRARIES} )
elseif( Cobalt_BACKEND MATCHES "HIP")
  message( STATUS "CobaltLib linking HIP: ${HCC_INCLUDE_DIRS}" )
  include_directories( ${HCC_INCLUDE_DIRS} )
  target_link_libraries( ${CobaltLib_NAME} ${HSA_LIBRARIES} )
endif()


