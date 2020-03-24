cmake_minimum_required(VERSION 3.14)
project(PeopleCounter)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_INCLUDE_DIRS "/usr/local/Cellar/opencv/4.2.0_3/include/opencv4;/usr/local/Cellar/opencv/4.2.0_3/include/opencv4")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

add_executable(PeopleCounter main.cpp)

target_link_libraries(PeopleCounter PRIVATE ${OpenCV_LIBS})