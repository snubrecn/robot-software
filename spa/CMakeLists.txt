cmake_minimum_required(VERSION 3.2.0)

project(spa)

# enable_testing()
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTest REQUIRED)

include_directories(
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${GTEST_INCLUDE_DIR})

add_executable(${PROJECT_NAME}_test spa_test.cc)
target_link_libraries(${PROJECT_NAME}_test
    ${EIGEN_LIBRARIES}
    ${CERES_LIBRARIES}
    ${GTEST_BOTH_LIBRARIES})
