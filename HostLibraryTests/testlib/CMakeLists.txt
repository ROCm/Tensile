
set(testlib_sources
    source/TestData.cpp)

add_library(TensileTestLib STATIC ${testlib_sources})

target_include_directories(TensileTestLib PUBLIC include)
target_link_libraries(TensileTestLib PUBLIC Boost::filesystem)

