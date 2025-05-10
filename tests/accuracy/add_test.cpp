// test_main.cpp
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "cpu/kernels/cpu_kernels.h"
#include "gpu/kernels/gpu_kernels.h"
#include "common/tensor/tensor.h"
#include "utils/random.h"
// #include <cupti.h>
using namespace utils;
using namespace common;

// void CUPTIAPI callbackHandler(void *userdata, CUpti_CallbackDomain domain,
//                               CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
//     if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
//         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
//         printf("CUDA kernel launch: %s\n", cbInfo->symbolName);
//     }
//     std::cout << "domain id: " << domain << ", callback id: " << cbid << std::endl;
//     if (cbInfo->functionName)
//         std::cout << "function name : " << ((CUpti_CallbackData*)cbInfo)->functionName << std::endl;
// }

TEST(AddAccTest, _2560_x_8192) {
    std::vector<int64_t> shape{2, 16};
    Tensor lhs = gen_rand_tensor(shape);
    Tensor rhs = gen_rand_tensor(shape);
    Tensor cpu_result(shape);
    Tensor gpu_result(shape, DataType::FLOAT32, DeviceType::GPU);

    // CUpti_SubscriberHandle subscriber;
    // // 订阅 Runtime API 回调
    // cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callbackHandler, nullptr);
    // // cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
    // // cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    // cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);

    cpu_kernels::add(lhs.data_ptr<float>(), rhs.data_ptr<float>(), cpu_result.data_ptr<float>(), cpu_result.elements_num());

    auto lhs_gpu = lhs.to(DeviceType::GPU);
    auto rhs_gpu = rhs.to(DeviceType::GPU);
    gpu_kernels::add(lhs_gpu.data_ptr<float>(), rhs_gpu.data_ptr<float>(), gpu_result.data_ptr<float>(), gpu_result.elements_num());

    auto ret = gpu_result.to(DeviceType::CPU);
    for (int64_t index = 0; index < lhs.elements_num(); index++) {
        if (cpu_result.data_ptr<float>()[index] != ret.data_ptr<float>()[index]) {
            std::cout << "error index: " << index << std::endl;
            std::cout << "cpu_result: " << cpu_result.data_ptr<float>()[index] << std::endl;
            std::cout << "gpu_result: " << ret.data_ptr<float>()[index] << std::endl;
            ASSERT_TRUE(false);
        }
    }
}
