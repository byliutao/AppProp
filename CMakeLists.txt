cmake_minimum_required(VERSION 3.25)
project(AppProp)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )

add_executable(AppProp main.cpp AppProp.h AppProp.cpp)

target_link_libraries(AppProp ${OpenCV_LIBS})

target_include_directories(AppProp PUBLIC
        "${OpenCV_INCLUDE_DIRS}"
        "${EIGEN3_INCLUDE_DIRS}"
        )