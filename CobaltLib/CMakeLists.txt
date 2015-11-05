
# CobaltLib public API headers
source_group(public_headers  FILES ${CobaltLib_PUBLIC_HEADERS} )

# CobaltLib source files
set( CobaltLib_SRC
  src/AssignSolution.h
  src/Logger.h
  src/Solution.h

  src/Control.cpp
  src/Dependency.cpp
  src/Device.cpp
  src/Operation.cpp
  src/Problem.cpp
  src/Status.cpp
  src/Tensor.cpp

  src/AssignSolution.cpp
  src/Logger.cpp
  src/Solution.cpp
  )
source_group(source FILES ${CobaltLib_SRC} )

# Cobalt include directories
include_directories(
  include
  src
  )

# build library
add_library( ${CobaltLib_NAME}
  ${CobaltLib_PUBLIC_HEADERS}
  ${CobaltLib_SRC}
  )

add_definitions(-DLOG_FILE_PREFIX="${PROJECT_BINARY_DIR}/cobalt_log.xml")
