add_executable(
  gcel_tests
  test_main.cpp
  add_test.cpp
  transpose_test.cpp
  gemm_test.cpp
)

target_link_libraries(
  gcel_tests
  gtest_main
  CPU_KERNELS
  GPU_OP
  UTILS
  Tensor
)

# include(GoogleTest)
gtest_discover_tests(gcel_tests)
