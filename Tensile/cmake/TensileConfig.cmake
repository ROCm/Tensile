################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

include(CMakeParseArguments)

if(NOT DEFINED Tensile_ROOT)
    # Compute the installation prefix relative to this file.
    get_filename_component(Tensile_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
    get_filename_component(Tensile_PREFIX "${Tensile_PREFIX}" PATH)

    if (WIN32)
        execute_process(COMMAND "${Tensile_PREFIX}/bin/TensileGetPath.exe" OUTPUT_VARIABLE Tensile_ROOT)
    else()
        execute_process(COMMAND "${Tensile_PREFIX}/bin/TensileGetPath" OUTPUT_VARIABLE Tensile_ROOT)
    endif()
endif()
list(APPEND CMAKE_MODULE_PATH "${Tensile_ROOT}/Source/cmake/")
list(APPEND CMAKE_MODULE_PATH "${Tensile_ROOT}/Source/")

if("HIP" IN_LIST Tensile_FIND_COMPONENTS)
    set(TENSILE_USE_HIP ON CACHE BOOL "Use HIP")
else()
    set(TENSILE_USE_HIP OFF CACHE BOOL "Use HIP")
endif()

if("LLVM" IN_LIST Tensile_FIND_COMPONENTS)
    set(TENSILE_USE_LLVM ON CACHE BOOL "Use LLVM")
else()
    set(TENSILE_USE_LLVM OFF CACHE BOOL "Use LLVM")
endif()

if("Client" IN_LIST Tensile_FIND_COMPONENTS)
    if(TENSILE_USE_HIP AND TENSILE_USE_LLVM)
        set(TENSILE_BUILD_CLIENT ON CACHE BOOL "Build Client")
    elseif(Tensile_FIND_REQUIRED_Client)
        message("Tensile client requires both Hip and LLVM.")
        set(Tensile_FOUND false)
    else()
        set(TENSILE_BUILD_CLIENT OFF CACHE BOOL "Build Client")
    endif()
else()
    set(TENSILE_BUILD_CLIENT OFF CACHE BOOL "Build Client")
endif()

if("STATIC_ONLY" IN_LIST Tensile_FIND_COMPONENTS)
    set(TENSILE_STATIC_ONLY ON CACHE BOOL "Disable exporting symbols from shared library.")
else()
    set(TENSILE_STATIC_ONLY OFF CACHE BOOL "Disable exporting symbols from shared library.")
endif()

add_subdirectory("${Tensile_ROOT}/Source" "Tensile")
include("${Tensile_ROOT}/Source/TensileCreateLibrary.cmake")

# Target is created for copying dependencies
function(TensileCreateCopyTarget
    Target_NAME
    Tensile_OBJECTS_TO_COPY
    Dest_PATH
    )

    file(MAKE_DIRECTORY "${Dest_PATH}")
    add_custom_target(
        ${Target_NAME} ALL
        COMMENT "${Target_NAME}: Copying tensile objects to ${Dest_PATH}"
        COMMAND_EXPAND_LISTS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Tensile_OBJECTS_TO_COPY} ${Dest_PATH}
        DEPENDS ${Tensile_OBJECTS_TO_COPY}
    )
endfunction()

# Output target: ${Tensile_VAR_PREFIX}_LIBRARY_TARGET. Ensures that the libs get built in Tensile_OUTPUT_PATH/library.
# Output symbol: ${Tensile_VAR_PREFIX}_ALL_FILES. List of full paths of all expected library files in manifest.
function(TensileCreateLibraryFiles
         Tensile_LOGIC_PATH
         Tensile_OUTPUT_PATH
         )

  if(NOT TENSILE_NEW_CLIENT)
    message(FATAL_ERROR "TensileCreateLibraryFiles function should only be called for new client.")
  endif()

  # Boolean options
  set(options
       MERGE_FILES
       NO_MERGE_FILES
       SHORT_FILE_NAMES
       PRINT_DEBUG
       GENERATE_PACKAGE
       )

  # Single value settings
  set(oneValueArgs
       CODE_OBJECT_VERSION
       COMPILER
       COMPILER_PATH
       EMBED_KEY
       EMBED_LIBRARY
       LIBRARY_FORMAT
       TENSILE_ROOT
       VAR_PREFIX
       )

  # Multi value settings
  set(multiValueArgs
       ARCHITECTURE
       )

  cmake_parse_arguments(Tensile "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(Tensile_UNPARSED_ARGUMENTS)
    message(WARNING "Unrecognized arguments: ${Tensile_UNPARSED_ARGUMENTS}")
  endif()
  if(Tensile_KEYWORDS_MISSING_VALUES)
    message(WARNING "Malformed arguments: ${Tensile_KEYWORDS_MISSING_VALUES}")
  endif()

  # Parse incoming options
  if(Tensile_TENSILE_ROOT)
    set(Script "${Tensile_TENSILE_ROOT}/bin/TensileCreateLibrary")
  else()
    set(Script "${Tensile_ROOT}/bin/TensileCreateLibrary")
  endif()

  message(STATUS "Tensile script: ${Script}")

  set(Options "--new-client-only" "--no-legacy-components")

  # Older NO_MERGE_FILES flag overrides MERGE_FILES option.
  if(Tensile_NO_MERGE_FILES)
    set(Tensile_MERGE_FILES FALSE)
  endif()

  if(Tensile_MERGE_FILES)
    set(Options ${Options} "--merge-files")
  else()
    set(Options ${Options} "--no-merge-files")
  endif()

  if(Tensile_GENERATE_PACKAGE)
    set(Options ${Options} "--package-library")
  endif()

  if(Tensile_SHORT_FILE_NAMES)
    set(Options ${Options} "--short-file-names")
  else()
    set(Options ${Options} "--no-short-file-names")
  endif()

  if(Tensile_PRINT_DEBUG)
    set(Options ${Options} "--library-print-debug")
  else()
    set(Options ${Options} "--no-library-print-debug")
  endif()

  if(Tensile_EMBED_LIBRARY)
    set(Options ${Options} "--embed-library=${Tensile_EMBED_LIBRARY}")
  endif()

  if(Tensile_EMBED_KEY)
    set(Options ${Options} "--embed-library-key=${Tensile_EMBED_KEY}")
  endif()

  if(Tensile_CODE_OBJECT_VERSION)
    set(Options ${Options} "--code-object-version=${Tensile_CODE_OBJECT_VERSION}")
  endif()

  if(Tensile_COMPILER)
    set(Options ${Options} "--cxx-compiler=${Tensile_COMPILER}")
  endif()

  if(Tensile_COMPILER_PATH)
    set(Options ${Options} "--cmake-cxx-compiler=${Tensile_COMPILER_PATH}")
  endif()

  if(Tensile_LIBRARY_FORMAT)
    set(Options ${Options} "--library-format=${Tensile_LIBRARY_FORMAT}")
    if(Tensile_LIBRARY_FORMAT MATCHES "yaml")
        target_compile_definitions( TensileHost PUBLIC -DTENSILE_YAML=1)
    endif()
  endif()
  
  if(Tensile_ARCHITECTURE)
    set(ListOptions "--architecture=${Tensile_ARCHITECTURE}")
  else()
    set(ListOptions "")
  endif()

  if (WIN32)
    set(Script ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} ${Script} ${Options})
  else()
    set(Script ${Script} ${Options})
  endif()
  set(PathArgs ${Tensile_LOGIC_PATH} ${Tensile_OUTPUT_PATH} "HIP")
  message(STATUS "Tensile_CREATE_COMMAND: ${Script} ${ListOptions} ${PathArgs}")

  if(Tensile_EMBED_LIBRARY)
      set(Tensile_EMBED_LIBRARY_SOURCE "${Tensile_OUTPUT_PATH}/library/${Tensile_EMBED_LIBRARY}.cpp")
  endif()

  if($ENV{TENSILE_SKIP_LIBRARY})
      message(STATUS "Skipping build of ${Tensile_OUTPUT_PATH}")
  else()

      if(NOT Tensile_VAR_PREFIX)
          set(Tensile_VAR_PREFIX TENSILE)
      endif()

      set(Tensile_MANIFEST_FILE_PATH "${Tensile_OUTPUT_PATH}/library/TensileManifest.txt")
      message(STATUS "Tensile_MANIFEST_FILE_PATH: ${Tensile_MANIFEST_FILE_PATH}")

      # Create the manifest file of the output libraries.
      execute_process(
        COMMAND ${Script} "${ListOptions}" ${PathArgs} "--generate-manifest-and-exit"
        RESULT_VARIABLE Tensile_CREATE_MANIFEST_RESULT
        COMMAND_ECHO STDOUT)

      if(Tensile_CREATE_MANIFEST_RESULT OR (NOT EXISTS ${Tensile_MANIFEST_FILE_PATH}))
        message(FATAL_ERROR "Error creating Tensile library: ${Tensile_CREATE_MANIFEST_RESULT}")
      endif()

      # Defer the actual call of the TensileCreateLibraries to 'make' time as needed.
      # Read the manifest file and declare the files as expected output.
      file(STRINGS ${Tensile_MANIFEST_FILE_PATH} Tensile_MANIFEST_CONTENTS)
      add_custom_command(
        COMMENT "Generating Tensile Libraries"
        OUTPUT ${Tensile_EMBED_LIBRARY_SOURCE};${Tensile_MANIFEST_CONTENTS}
        COMMAND ${Script} "${ListOptions}" ${PathArgs} 
      )

      set("${Tensile_VAR_PREFIX}_ALL_FILES" ${Tensile_MANIFEST_CONTENTS} PARENT_SCOPE)

      # Create a chained library build target.
      # We've declared the manifest contents as output of the custom
      # command above which builds the tensile libs. Now create a
      # target dependency on those files so that we force the custom
      # command to be invoked at build time, not cmake time.
      TensileCreateCopyTarget(
      "${Tensile_VAR_PREFIX}_LIBRARY_TARGET"
      "${Tensile_MANIFEST_CONTENTS}"
      "${Tensile_OUTPUT_PATH}/library"
    )

  endif()

  if(Tensile_EMBED_LIBRARY)

    add_library(${Tensile_EMBED_LIBRARY} ${Tensile_EMBED_LIBRARY_SOURCE})
    target_link_libraries(${Tensile_EMBED_LIBRARY} PUBLIC TensileHost)

  endif()

endfunction()

