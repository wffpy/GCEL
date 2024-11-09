add_library(GPU_OP STATIC add.cu transpose.cu)
set_target_properties(GPU_OP PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(GPU_OP ${CUDA_LIBRARIES})