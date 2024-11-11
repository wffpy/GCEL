add_executable(
  gcel_tests
  test_main.cpp
  add_test.cpp
  transpose_test.cpp
)

target_link_libraries(
  gcel_tests
  gtest_main
  CPU_OP
  GPU_OP
  UTILS
)

# include(GoogleTest)
gtest_discover_tests(gcel_tests)
