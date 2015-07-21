include_directories( "../../include" "/usr/local/cuda/include" ${GLEW_INCLUDE_DIRS} SYSTEM ${INCLUDE_DIRS})
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
cuda_add_library(libgpunbody STATIC bodySystem.cu
                 OPTIONS
                 DEBUG          "-std=c++11 -g -G -O0 -lineinfo -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 "
                 RELEASE        "-std=c++11 -g    -O3 -lineinfo -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 --ptxas-options -v "
                 MINSIZEREL     "-std=c++11 -g    -O3 -lineinfo -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 "
                 RELWITHDEBINFO "-std=c++11 -g    -O3 -lineinfo -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 " )