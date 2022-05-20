find_package(stb REQUIRED)
find_package(spdlog REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_BINARY_DIR}/config/include")
target_link_libraries(${PROJECT_NAME} stb::stb spdlog::spdlog ncnn)


