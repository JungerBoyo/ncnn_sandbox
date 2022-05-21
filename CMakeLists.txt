cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_CXX_STANDARD 20)

project(ncnn_sandbox VERSION 0.0.1 LANGUAGES C CXX )

# include(CTest)
# enable_testing()

execute_process(COMMAND conan install ${CMAKE_CURRENT_SOURCE_DIR} 
  --install-folder ${CMAKE_BINARY_DIR}
  -pr=${CMAKE_CURRENT_SOURCE_DIR}/conan.profile
)

set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR} ${CMAKE_PREFIX_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(config)
add_subdirectory(res)
add_subdirectory(src)

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
