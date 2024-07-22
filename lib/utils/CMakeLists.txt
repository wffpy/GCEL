add_subdirectory(tensor)
add_library(UTILS STATIC random.cpp)

target_link_libraries(UTILS Tensor)

