
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

set(client_sources
    DataInitialization.cpp
    ClientProblemFactory.cpp
    main.cpp)

find_package(Boost COMPONENTS program_options)

add_library(TensileClient OBJECT ${client_sources})

target_include_directories(TensileClient PUBLIC Tensile ${Boost_INCLUDE_DIRS})

add_executable(tensile $<TARGET_OBJECTS:TensileClient>)

target_link_libraries(tensile Tensile ${Boost_LIBRARIES})


