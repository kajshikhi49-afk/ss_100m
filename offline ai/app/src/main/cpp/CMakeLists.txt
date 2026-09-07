cmake_minimum_required(VERSION 3.22.1)
project(bengali_ai C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable unnecessary llama.cpp subprojects
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama: build tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama: build examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama: build server" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "llama: build common" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "build shared libraries" FORCE)

# Path to llama.cpp root
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../llama.cpp-master)

# Include llama.cpp subdirectories
add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp-build EXCLUDE_FROM_ALL)

# JNI bridge library
add_library(bengali_ai_jni SHARED
    jni_bridge.cpp
)

target_include_directories(bengali_ai_jni PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_libraries(bengali_ai_jni PRIVATE
    llama
    ggml
    ggml-base
    log
    android
)
