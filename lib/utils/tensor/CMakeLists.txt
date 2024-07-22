add_library(Tensor STATIC tensor.cpp 
                TensorImpl.cpp 
                CpuTensorImpl.cpp
                GpuTensorImpl.cpp
                basic.cpp)
