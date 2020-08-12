cmake_minimum_required(VERSION 3.17)
project(simplebf)

set(CMAKE_CXX_STANDARD 17)
find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")



add_subdirectory(PEGTL)
include_directories(cxxopts/include)
add_executable(simplebf main.cpp)
target_link_libraries(simplebf taocpp::pegtl LLVM)
set_property(TARGET "simplebf" APPEND_STRING PROPERTY LINK_FLAGS "${LLVM_LINK_FLAGS}")