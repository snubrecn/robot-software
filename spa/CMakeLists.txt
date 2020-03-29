cmake_minimum_required(VERSION 3.2.0)

project(spa)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_executable(${PROJECT_NAME} main.cc)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ${CERES_LIBRARIES})
