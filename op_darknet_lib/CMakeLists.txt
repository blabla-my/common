cmake_minimum_required(VERSION 2.8.3)
project(op_darknet_lib)

# when use the following configuration, the op_utility function GetTickCount doesn't link. 
#if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
#elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,error")
#endif()

# Enable support for C++14
if(${CMAKE_VERSION} VERSION_LESS "3.1.0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
else()
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge    
  roscpp    
  roslint
  image_transport           
  op_planner
  op_ros_helpers     
)

set(
    CUDA_NVCC_FLAGS    
    "-O3        
    -gencode arch=compute_35,code=sm_35 
    -gencode arch=compute_50,code=[sm_50,compute_50] 
    -gencode arch=compute_52,code=[sm_52,compute_52]    
    -gencode arch=compute_61,code=[sm_61,compute_61] 
    -gencode arch=compute_70,code=[sm_70,compute_70] 
    -gencode arch=compute_86,code=[sm_86,compute_86] "    
    )

add_definitions(-DGPU -DCUDNN -DCUDNN_HALF -DOPENCV -fopenmp -lgomp -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a)


catkin_package(
  INCLUDE_DIRS include 
  LIBRARIES ${PROJECT_NAME}  
  CATKIN_DEPENDS roscpp roslint cv_bridge image_transport op_planner op_ros_helpers
  DEPENDS 
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  include/${PROJECT_NAME} include/${PROJECT_NAME}/darknet ${OpenCV_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS} /usr/local/cuda-11.1/include 
)

link_directories(    
  /usr/local/cuda-11.1/lib64
)

set(DARKNET_SRC_FILES
	src/DarknetDetector.cpp	
	src/dummydetector2.cpp
	src/darknet/activation_layer.c
	src/darknet/activations.c
	src/darknet/art.c
	src/darknet/avgpool_layer.c
	src/darknet/batchnorm_layer.c
	src/darknet/blas.c
	src/darknet/box.c
	src/darknet/captcha.c
	src/darknet/cifar.c
	src/darknet/classifier.c
	src/darknet/coco.c
	src/darknet/col2im.c
	src/darknet/compare.c
	src/darknet/connected_layer.c
	src/darknet/conv_lstm_layer.c
	src/darknet/convolutional_layer.c
	src/darknet/cost_layer.c
	src/darknet/cpu_gemm.c
	src/darknet/crnn_layer.c
	src/darknet/crop_layer.c
	src/darknet/dark_cuda.c	
	src/darknet/data.c
	src/darknet/deconvolutional_layer.c
	src/darknet/demo.c
	src/darknet/detection_layer.c
	src/darknet/detector.c
	src/darknet/dice.c
	src/darknet/dropout_layer.c
	src/darknet/gaussian_yolo_layer.c
	src/darknet/gemm.c
	src/darknet/getopt.c
	src/darknet/gettimeofday.c
	src/darknet/go.c
	src/darknet/gru_layer.c
	src/darknet/http_stream.cpp
	src/darknet/im2col.c
	src/darknet/image_opencv.cpp
	src/darknet/image.c
	src/darknet/layer.c
	src/darknet/list.c
	src/darknet/local_layer.c
	src/darknet/lstm_layer.c
	src/darknet/matrix.c
	src/darknet/maxpool_layer.c
	src/darknet/network.c
	src/darknet/nightmare.c
	src/darknet/normalization_layer.c
	src/darknet/option_list.c
	src/darknet/parser.c
	src/darknet/region_layer.c	
	src/darknet/reorg_layer.c
	src/darknet/reorg_old_layer.c
	src/darknet/rnn_layer.c
	src/darknet/rnn_vid.c
	src/darknet/route_layer.c
	src/darknet/sam_layer.c
	src/darknet/scale_channels_layer.c
	src/darknet/shortcut_layer.c
	src/darknet/softmax_layer.c
	src/darknet/super.c	
	src/darknet/swag.c
	src/darknet/tag.c
	src/darknet/tree.c
	src/darknet/upsample_layer.c		
	src/darknet/utils.c	
	src/darknet/voxel.c
	src/darknet/writing.c
	src/darknet/yolo_layer.c		
	src/darknet/yolo_v2_class.cpp
	src/darknet/yolo.c
	
	src/darknet/activation_kernels.cu
	src/darknet/avgpool_layer_kernels.cu
	src/darknet/blas_kernels.cu
	src/darknet/col2im_kernels.cu
	src/darknet/convolutional_kernels.cu
	src/darknet/crop_layer_kernels.cu
	src/darknet/deconvolutional_kernels.cu
	src/darknet/dropout_layer_kernels.cu
	src/darknet/im2col_kernels.cu
	src/darknet/maxpool_layer_kernels.cu
	src/darknet/network_kernels.cu	
)

## Declare a cpp library
cuda_add_library(${PROJECT_NAME}
  ${DARKNET_SRC_FILES}
)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBRARIES}
	${CUDA_LIBRARIES}
	cudart
   	cublas
   	curand
   	gomp
   	cudnn
   	cuda
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
