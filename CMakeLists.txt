cmake_minimum_required(VERSION 3.10)
project(vi_init_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ruta a Eigen
find_package(Eigen3 REQUIRED)

# Incluir headers
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Archivos fuente
add_executable(vi_init
    src/main.cpp
    src/bg_small_angle.cpp
)
