cmake_minimum_required(VERSION 2.8.3)
project(dl_interface)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  dark_msgs
)

find_package(CUDA)
find_package(OpenCV REQUIRED)
find_package(OpenMP)

catkin_package(
        CATKIN_DEPENDS cv_bridge
        image_transport
        roscpp
        sensor_msgs
        std_msgs
        dark_msgs
)


set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -Wall ${CMAKE_CXX_FLAGS}")

IF (CUDA_FOUND)
    list(APPEND CUDA_NVCC_FLAGS "--std=c++11 -I$${PROJECT_SOURCE_DIR}/darknet/src -I${PROJECT_SOURCE_DIR}/src -DGPU")
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
    message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
    set(
      CUDA_NVCC_FLAGS
      ${CUDA_NVCC_FLAGS};
      -O3
      -gencode arch=compute_30,code=sm_30
      -gencode arch=compute_35,code=sm_35
      -gencode arch=compute_50,code=[sm_50,compute_50]
      -gencode arch=compute_52,code=[sm_52,compute_52]
      -gencode arch=compute_61,code=sm_61
      -gencode arch=compute_62,code=sm_62
    )
    add_definitions(-DGPU)
    #darknet
    cuda_add_library(darknet_roslib SHARED
            darknet/src/activation_kernels.cu
            darknet/src/avgpool_layer_kernels.cu
            darknet/src/convolutional_kernels.cu
            darknet/src/crop_layer_kernels.cu
            darknet/src/col2im_kernels.cu
            darknet/src/blas_kernels.cu
            darknet/src/deconvolutional_kernels.cu
            darknet/src/dropout_layer_kernels.cu
            darknet/src/im2col_kernels.cu
            darknet/src/maxpool_layer_kernels.cu

            darknet/src/gemm.c
            darknet/src/utils.c
            darknet/src/cuda.c
            darknet/src/deconvolutional_layer.c
            darknet/src/convolutional_layer.c
            darknet/src/list.c
            darknet/src/image.c
            darknet/src/activations.c
            darknet/src/im2col.c
            darknet/src/col2im.c
            darknet/src/blas.c
            darknet/src/crop_layer.c
            darknet/src/dropout_layer.c
            darknet/src/maxpool_layer.c
            darknet/src/softmax_layer.c
            darknet/src/data.c
            darknet/src/matrix.c
            darknet/src/network.c
            darknet/src/connected_layer.c
            darknet/src/cost_layer.c
            darknet/src/parser.c
            darknet/src/option_list.c
            darknet/src/detection_layer.c
            darknet/src/route_layer.c
            darknet/src/upsample_layer.c
            darknet/src/box.c
            darknet/src/normalization_layer.c
            darknet/src/avgpool_layer.c
            darknet/src/layer.c
            darknet/src/local_layer.c
            darknet/src/shortcut_layer.c
            darknet/src/logistic_layer.c
            darknet/src/activation_layer.c
            darknet/src/rnn_layer.c
            darknet/src/gru_layer.c
            darknet/src/crnn_layer.c
            darknet/src/batchnorm_layer.c
            darknet/src/region_layer.c
            darknet/src/reorg_layer.c
            darknet/src/tree.c
            darknet/src/lstm_layer.c
            darknet/src/l2norm_layer.c
            darknet/src/yolo_layer.c
            )

    target_compile_definitions(darknet_roslib PUBLIC -DGPU)
    cuda_add_cublas_to_target(darknet_roslib)

    if (OPENMP_FOUND)
        set_target_properties(darknet_roslib PROPERTIES
                COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
                LINK_FLAGS ${OpenMP_CXX_FLAGS}
                )
    endif ()

    target_include_directories(darknet_roslib PRIVATE
            ${OpenCV_INCLUDE_DIR}
            ${catkin_INCLUDE_DIRS}
            ${Boost_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            ${PROJECT_SOURCE_DIR}/darknet
            ${PROJECT_SOURCE_DIR}/darknet/src
            ${PROJECT_SOURCE_DIR}/src
            )

    target_link_libraries(darknet_roslib
            ${OpenCV_LIBRARIES}
            ${catkin_LIBRARIES}
            ${PCL_LIBRARIES}
            ${Qt5Core_LIBRARIES}
            ${CUDA_LIBRARIES}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_curand_LIBRARY}
            )

    add_dependencies(darknet_roslib
            ${catkin_EXPORTED_TARGETS}
            )

    #ros node
    cuda_add_executable(dark_ros
            src/dark_ros.cpp
            src/dark_yolo.h
            )

    target_compile_definitions(dark_ros PUBLIC -DGPU)

    target_include_directories(dark_ros PRIVATE
            ${CUDA_INCLUDE_DIRS}
            ${catkin_INCLUDE_DIRS}
            ${PROJECT_SOURCE_DIR}/darknet
            ${PROJECT_SOURCE_DIR}/darknet/src
            ${PROJECT_SOURCE_DIR}/src
            )

    target_link_libraries(dark_ros
            ${catkin_LIBRARIES}
            ${OpenCV_LIBS}
            ${CUDA_LIBRARIES}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_curand_LIBRARY}
            darknet_roslib
            )
    add_dependencies(dark_ros
            ${catkin_EXPORTED_TARGETS}
            )
    install(TARGETS darknet_roslib dark_ros
            ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
            )
ELSE()
    message("CUDA Not found. The package won't be built")
ENDIF ()
