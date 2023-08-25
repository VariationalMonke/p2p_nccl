cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(nccl_bw LANGUAGES CXX CUDA)

find_path(NCCL_INCLUDE_DIRS nccl.h)
find_library(NCCL_LIBRARIES nccl)

add_executable(nccl_bw nccl_bw.cu)
target_include_directories(nccl_bw PRIVATE ${NCCL_INCLUDE_DIRS})
target_link_libraries(nccl_bw ${NCCL_LIBRARIES})