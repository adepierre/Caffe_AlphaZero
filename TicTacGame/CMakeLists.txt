cmake_minimum_required(VERSION 3.1)
project(TicTacGame)

set(SRC_FILES 
${PROJECT_SOURCE_DIR}/src/main.cpp
)

set(HDR_FILES 
${PROJECT_SOURCE_DIR}/include/TicTacState.h
)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(TicTacGame ${SRC_FILES} ${HDR_FILES})
target_link_libraries(TicTacGame AlphaZero)