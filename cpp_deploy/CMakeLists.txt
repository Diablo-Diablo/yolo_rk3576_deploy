cmake_minimum_required(VERSION 3.10)
project(yolo11_rknn_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_library(RKNN_RT_LIB rknnrt PATHS /usr/lib /usr/local/lib REQUIRED)

include_directories(
    /usr/include
    ${OpenCV_INCLUDE_DIRS}
)

# 静态图像单次推理 Demo
add_executable(yolo11_demo main.cpp)
target_link_libraries(yolo11_demo ${OpenCV_LIBS} ${RKNN_RT_LIB} pthread)

# 实时摄像头采集与推理程序
add_executable(camera_yolo11 camera_yolo11.cpp)
target_link_libraries(camera_yolo11 ${OpenCV_LIBS} ${RKNN_RT_LIB} pthread)
