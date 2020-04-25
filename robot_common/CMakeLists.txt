cmake_minimum_required(VERSION 3.2.0)

project(robot_common VERSION 0.1)

set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS )
set(LIB ${PROJECT_NAME})

find_package(Eigen3 REQUIRED)

file(GLOB_RECURSE LIB_SRCS "src/*.cc")
file(GLOB_RECURSE LIB_HDRS "include/*.h")

include_directories(include ${EIGEN3_INCLUDE_DIR})
configure_file(include/version.h.in ${PROJECT_SOURCE_DIR}/include/version.h)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
add_library(${LIB} ${LIB_SRCS} ${LIB_HDRS})
target_link_libraries(${LIB} ${EIGEN_LIBRARIES})
