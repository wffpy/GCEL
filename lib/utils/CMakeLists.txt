add_subdirectory(tensor)
add_subdirectory(log)

add_library(UTILS STATIC random.cpp test_helper.cpp)
target_link_libraries(UTILS Tensor Log)

