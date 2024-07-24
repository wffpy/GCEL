cmake_minimum_required(VERSION 3.10)
project(GCELProject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)  # 对应于 A800 架构

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 检测并设置AVX编译选项
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_MAVX)
if(COMPILER_SUPPORTS_MAVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
else()
    message(FATAL_ERROR "The compiler does not support AVX extensions")
endif()


# find_package(CUDA REQUIRED)
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(include)
add_subdirectory(lib)


# 下载和解压 googletest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# 为了在本地不重新下载，先使用它
FetchContent_MakeAvailable(googletest)

# 或者只在没有找到的情况下重新下载
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

enable_testing()

add_executable(
  gcel_tests
  tests/test_main.cpp
  tests/add_test.cpp
)

target_link_libraries(
  gcel_tests
  gtest_main
  CPU_OP
  GPU_OP
  UTILS
)

include(GoogleTest)
gtest_discover_tests(gcel_tests)
