
################################################################################
# Global Options
################################################################################
cmake_minimum_required(VERSION 2.8)

########################################
# Cobalt project name
project( Cobalt C CXX )

########################################
# Cobalt Version
set( Cobalt_VERSION_MAJOR 0 )
set( Cobalt_VERSION_MINOR 1 )
set( Cobalt_VERSION_PATCH 0 )

########################################
# Cobalt Mode
set( Cobalt_MODE "Cobalt_MODE_LOG_ONLY" CACHE STRING
  "Cobalt operation mode?" )
set_property( CACHE Cobalt_MODE PROPERTY STRINGS
  Cobalt_MODE_LOG_ONLY
  Cobalt_MODE_SOLVE_ONLY
  Cobalt_MODE_LOG_AND_SOLVE
  Cobalt_MODE_NONE )
if(Cobalt_MODE MATCHES "Cobalt_MODE_LOG_ONLY" )
  add_definitions( -DCobalt_LOGGER_ENABLED=1)
  add_definitions( -DCobalt_SOLVER_ENABLED=0)
elseif(Cobalt_MODE MATCHES "Cobalt_MODE_SOLVE_ONLY" )
  add_definitions( -DCobalt_LOGGER_ENABLED=0)
  add_definitions( -DCobalt_SOLVER_ENABLED=1)
elseif(Cobalt_MODE MATCHES "Cobalt_MODE_LOG_AND_SOLVE" )
  add_definitions( -DCobalt_LOGGER_ENABLED=1)
  add_definitions( -DCobalt_SOLVER_ENABLED=1)
elseif(Cobalt_MODE MATCHES "Cobalt_MODE_NONE" )
  add_definitions( -DCobalt_LOGGER_ENABLED=0)
  add_definitions( -DCobalt_SOLVER_ENABLED=0)
endif()

########################################
# Cobalt Backend
set( Cobalt_BACKEND "OpenCL_2.0" CACHE STRING
  "Which of Cobalt's backends to use?" )
set_property( CACHE Cobalt_BACKEND PROPERTY STRINGS
  OpenCL_1.2 OpenCL_2.0 HCC HSA )
# OpenCL 1.2
if( Cobalt_BACKEND MATCHES "OpenCL_1.2")
  add_definitions( -DCobalt_BACKEND_OPENCL12=1 )
  add_definitions( -DCobalt_BACKEND_OPENCL20=0 )
  add_definitions( -DCobalt_BACKEND_HCC=0 )
  add_definitions( -DCobalt_BACKEND_HSA=0 )
  message( STATUS "${Cobalt_BACKEND} backend selected." )
endif()
# OpenCL 2.0
if( Cobalt_BACKEND MATCHES "OpenCL_2.0")
  add_definitions( -DCobalt_BACKEND_OPENCL12=0 )
  add_definitions( -DCobalt_BACKEND_OPENCL20=1 )
  add_definitions( -DCobalt_BACKEND_HCC=0 )
  add_definitions( -DCobalt_BACKEND_HSA=0 )
  message( STATUS "${Cobalt_BACKEND} backend selected." )
endif()
# HCC / Kalmar
if( Cobalt_BACKEND MATCHES "HCC")
  add_definitions( -DCobalt_BACKEND_OPENCL12=0 )
  add_definitions( -DCobalt_BACKEND_OPENCL20=0 )
  add_definitions( -DCobalt_BACKEND_HCC=1 )
  add_definitions( -DCobalt_BACKEND_HSA=0 )
  message( SEND_ERROR "${Cobalt_BACKEND} backend not yet supported." )
endif()
# HSA
if( Cobalt_BACKEND MATCHES "HSA")
  add_definitions( -DCobalt_BACKEND_OPENCL12=0 )
  add_definitions( -DCobalt_BACKEND_OPENCL20=0 )
  add_definitions( -DCobalt_BACKEND_HCC=0 )
  add_definitions( -DCobalt_BACKEND_HSA=1 )
  message( SEND_ERROR "${Cobalt_BACKEND} backend not yet supported." )
endif()

############################################################
# Cobalt Headers needed for all subprojects
set( CobaltLib_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/CobaltLib/include )
set( CobaltLib_PUBLIC_HEADERS ${CobaltLib_INCLUDE_DIR}/Cobalt.h )
set( CobaltLib_NAME CobaltLib )

################################################################################
# Cobalt Subprojects
################################################################################
add_subdirectory(CobaltLib)
add_subdirectory(clients)
