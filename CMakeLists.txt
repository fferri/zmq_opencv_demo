cmake_minimum_required(VERSION 2.6)

project(bluevision-rpi)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR 1)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-narrowing")

find_package(Boost REQUIRED COMPONENTS thread system regex timer filesystem serialization)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc features2d photo)
find_package(ZMQ REQUIRED)

include_directories(..)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${ZMQ_INCLUDE_DIRS})

set(
    LIBS
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ZMQ_LIBRARY}
)

add_executable(main main.cpp)
target_link_libraries(main ${LIBS})

