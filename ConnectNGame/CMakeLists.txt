cmake_minimum_required(VERSION 3.1)
project(ConnectNGame)

set(SRC_FILES 
${PROJECT_SOURCE_DIR}/src/main.cpp
)

set(HDR_FILES 
${PROJECT_SOURCE_DIR}/include/ConnectNState.h
)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(ConnectNGame ${SRC_FILES} ${HDR_FILES})
target_link_libraries(ConnectNGame AlphaZero)