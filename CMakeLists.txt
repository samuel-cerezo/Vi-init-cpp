cmake_minimum_required(VERSION 3.10)
project(vi_init_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

# Eigen (desde Homebrew)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# yaml-cpp desde Homebrew
set(YAML_CPP_DIR "/opt/homebrew/opt/yaml-cpp")
include_directories(${YAML_CPP_DIR}/include)
link_directories(${YAML_CPP_DIR}/lib)

# Archivos fuente
add_executable(vi_init
    src/main.cpp
    src/bg_small_angle.cpp
    src/euroc_io.cpp
)

# Enlazar con yaml-cpp
target_link_libraries(vi_init yaml-cpp)
