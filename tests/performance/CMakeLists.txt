add_executable(
  gcel_perf_tests
  test_main.cpp
  transpose_perf.cpp
)

target_link_libraries(
  gcel_perf_tests
  gtest_main
  CPU_KERNELS
  GPU_OP
  UTILS
  PROFILER
  Tensor
)

# include(GoogleTest)
gtest_discover_tests(gcel_perf_tests)
