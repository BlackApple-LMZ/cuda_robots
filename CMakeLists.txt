cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

project(cuda_robots)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDADevice REQUIRED)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

include_directories(${CUDA_INCLUDE_DIRS}) 
include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDADEVICE_NVCC_FLAGS} -Xptxas -v")

cuda_add_executable(basic_transformations
	src/basic_transformations/main.cpp
	src/basic_transformations/cudaWrapper.cpp
	src/basic_transformations/basic_transformations.cu
)
target_link_libraries(basic_transformations
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
)

