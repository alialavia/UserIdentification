cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project (externals)

add_subdirectory(${externals_SOURCE_DIR}/gflags)
add_subdirectory(${externals_SOURCE_DIR}/dlib)
