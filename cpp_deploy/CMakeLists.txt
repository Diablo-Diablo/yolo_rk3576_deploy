cmake_minimum_required(VERSION 3.10)
project(yolo11_rknn_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_library(RKNN_RT_LIB rknnrt PATHS /usr/lib /usr/local/lib REQUIRED)

add_executable(yolo11_demo main.cpp)
add_executable(camera_yolo11 camera_yolo11.cpp)
add_executable(camera_yolo11_face_fp16 camera_yolo11_face_fp16.cpp)
add_executable(camera_yolo11_face_int8 camera_yolo11_face_int8.cpp)

foreach(target IN ITEMS
    yolo11_demo
    camera_yolo11
    camera_yolo11_face_fp16
    camera_yolo11_face_int8
)
    target_include_directories(${target} PRIVATE
        /usr/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE
        ${OpenCV_LIBS}
        ${RKNN_RT_LIB}
        pthread
    )
endforeach()

