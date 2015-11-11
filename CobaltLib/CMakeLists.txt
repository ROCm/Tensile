
# CobaltLib public API headers
source_group(public_headers  FILES ${CobaltLib_PUBLIC_HEADERS} )

# CobaltLib source files
set( CobaltLib_SRC
  src/Cobalt.cpp
  src/Logger.cpp
  src/Logger.h
  src/ReferenceTensorContraction.cpp
  src/Solution.cpp
  src/Solution.h
  src/StructOperations.cpp
  src/StructOperations.h
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
