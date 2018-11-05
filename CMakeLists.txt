cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

project(opencv_playground)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.4.0 REQUIRED)

add_executable(opencv_test
    "${CMAKE_CURRENT_LIST_DIR}/opencv_test.cpp"
)

target_sources(opencv_test
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/opencv_test.h"
        "${CMAKE_CURRENT_LIST_DIR}/utils.h"
)

target_link_libraries(opencv_test
    PUBLIC
        "${OpenCV_LIBS}"
)


