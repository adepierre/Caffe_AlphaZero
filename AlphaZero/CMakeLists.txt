cmake_minimum_required(VERSION 3.1)
project(AlphaZero)

find_package(OpenCV)
find_package(Caffe)

set(HDR_FILES 
${PROJECT_SOURCE_DIR}/include/AlphaZero.h
${PROJECT_SOURCE_DIR}/include/GenericState.h
${PROJECT_SOURCE_DIR}/include/MCTS.h
${PROJECT_SOURCE_DIR}/include/Node.h
)

add_library(AlphaZero INTERFACE)
target_include_directories(AlphaZero INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(AlphaZero INTERFACE ${Caffe_LIBRARIES})

#Add a fake target to display AlphaZero files in Visual Studio
add_custom_target(AlphaZero_ide SOURCES ${HDR_FILES})