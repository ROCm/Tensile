
if(TENSILE_USE_LLVM)
    set(SOLUTION_LIBRARY_FILES_
        ${SOLUTION_LIBRARY_FILES_}
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLite.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLiteMixed.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLiteNavi.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsTileLite.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/SampleTensileKernels.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/rocBLAS_Full.yaml")
endif()

if(TENSILE_USE_MSGPACK)
    set(SOLUTION_LIBRARY_FILES_
        ${SOLUTION_LIBRARY_FILES_}
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLite.dat"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLiteMixed.dat"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLiteNavi.dat"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsTileLite.dat"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/SampleTensileKernels.dat"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/rocBLAS_Full.dat")
endif()

set(SOLUTION_LIBRARY_FILES
    ${SOLUTION_LIBRARY_FILES_}
    PARENT_SCOPE)
