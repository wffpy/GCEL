add_library(Tensor STATIC tensor.cpp 
                TensorImpl.cpp 
                CpuTensorImpl.cpp
                GpuTensorImpl.cpp
                MetaTensorImpl.cpp
                basic.cpp)
