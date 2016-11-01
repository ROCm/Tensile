
find_package(Cobalt CONFIG)

find_library(OPENCL_LIB OpenCL)

add_cobalt_lib(SimpleCobaltLib
    SOLUTIONS ${Cobalt_SOLUTIONS}
    BACKEND OpenCL_1.2
    
)
target_link_libraries(SimpleCobaltLib ${OPENCL_LIB})

add_definitions(-std=c++11)

add_executable(main main.cpp)
target_link_libraries(main SimpleCobaltLib)
