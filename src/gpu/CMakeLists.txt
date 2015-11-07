include_directories( "../../include" "/usr/local/cuda/include" ${GLEW_INCLUDE_DIRS} SYSTEM ${INCLUDE_DIRS})
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
cuda_add_library(libgpunbody STATIC bodySystem.cu
                 OPTIONS
                 DEBUG          "-std=c++11 -g -G -O0           -res-usage -lineinfo -gencode arch=compute_20,code=\"compute_20,sm_21\" -gencode arch=compute_30,code=\"compute_30,sm_35\" -gencode arch=compute_50,code=\"compute_50,sm_52\" "
                 RELEASE        "-std=c++11 -g    -O3 -restrict -res-usage -lineinfo -gencode arch=compute_20,code=\"compute_20,sm_21\" -gencode arch=compute_30,code=\"compute_30,sm_35\" -gencode arch=compute_50,code=\"compute_50,sm_52\" -maxrregcount=32 -use_fast_math"
                 MINSIZEREL     "-std=c++11       -O3 -restrict -res-usage -lineinfo -gencode arch=compute_20,code=\"compute_20,sm_21\" -gencode arch=compute_30,code=\"compute_30,sm_35\" -gencode arch=compute_50,code=\"compute_50,sm_52\" -maxrregcount=32 -use_fast_math"
                 RELWITHDEBINFO "-std=c++11 -g    -O3 -restrict -res-usage -lineinfo -gencode arch=compute_20,code=\"compute_20,sm_21\" -gencode arch=compute_30,code=\"compute_30,sm_35\" -gencode arch=compute_50,code=\"compute_50,sm_52\" -maxrregcount=32 -use_fast_math" )