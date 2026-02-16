cmake_minimum_required(VERSION 3.24)
project(MandelBrotOptim)

set(CMAKE_CXX_STANDARD 23)

# Use capitalized component names!
find_package(SFML COMPONENTS graphics window system REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(MandelBrotOptim main.cpp)

# Optimization for your Ryzen AI 350
target_compile_options(MandelBrotOptim PRIVATE -O3 -march=native -ffast-math -g)

# The "All-Keyword" signature to keep CMake happy
target_link_libraries(MandelBrotOptim
        PRIVATE
        sfml-graphics
        sfml-window
        sfml-system
        OpenMP::OpenMP_CXX
)
