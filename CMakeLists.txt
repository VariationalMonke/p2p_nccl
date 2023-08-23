cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 9)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(nccl_bw devices LANGUAGES CXX CUDA)

add_executable(nccl_bw nccl_bw.cu)