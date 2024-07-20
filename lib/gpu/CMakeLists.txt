#cuda_add_library(gpu_kernel add.cu)
#add_library(GPU_OP STATIC add.cpp add.cu)
add_library(GPU_OP STATIC add.cu)
set_target_properties(GPU_OP PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# target_link_libraries(GPU_OP gpu_kernel)
target_link_libraries(GPU_OP ${CUDA_LIBRARIES})