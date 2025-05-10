add_subdirectory(log)
add_subdirectory(cal_helper)

add_library(UTILS STATIC random.cpp test_helper.cpp)
target_link_libraries(UTILS Log CalHelper)

