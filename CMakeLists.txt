cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)

project(Caffe_AlphaZero)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_subdirectory(AlphaZero)
add_subdirectory(ConnectNGame)
add_subdirectory(TicTacGame)