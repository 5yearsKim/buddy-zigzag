cmake_minimum_required(VERSION 3.20)
project(BuddyZigzagGemmini LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Resolve MLIR symbols from the host buddy-opt, rather than linking a second
# copy of MLIR and its global pass/dialect registries into this plugin.
add_library(BuddyZigzagGemmini MODULE GemminiPasses.cpp)
target_include_directories(BuddyZigzagGemmini SYSTEM PRIVATE
  ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_options(BuddyZigzagGemmini PRIVATE ${LLVM_DEFINITIONS_LIST})
if(LLVM_ENABLE_ASSERTIONS)
  target_compile_options(BuddyZigzagGemmini PRIVATE -UNDEBUG)
endif()
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(BuddyZigzagGemmini PRIVATE -fno-rtti)
endif()
set_target_properties(BuddyZigzagGemmini PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
