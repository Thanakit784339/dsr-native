cmake_minimum_required(VERSION 3.22)
project(dsr_native LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# All HLE functions register themselves from static constructors, so the core is an OBJECT library:
# a static archive would let the linker drop "unreferenced" HLE translation units.
file(GLOB_RECURSE DSR_CORE_SOURCES CONFIGURE_DEPENDS
    src/common/*.cpp
    src/loader/*.cpp
    src/arch/*.cpp
    src/hle/*.cpp
    src/runtime/*.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" OR ANDROID_ABI STREQUAL "arm64-v8a")
    list(APPEND DSR_CORE_SOURCES src/arch/arm64/abi.S)
    set(DSR_ARM64 ON)
endif()

add_library(dsrcore OBJECT ${DSR_CORE_SOURCES})
target_include_directories(dsrcore PUBLIC src)
target_compile_options(dsrcore PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)
if(DSR_ARM64 AND NOT ANDROID)
    # Android reserves x18 by default; on other arm64 targets reserve it explicitly so host code
    # never depends on a value guest code may have clobbered.
    target_compile_options(dsrcore PUBLIC -ffixed-x18)
endif()

if(ANDROID)
    add_library(dsr SHARED
        src/platform/android/android_main.cpp
        ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
    target_include_directories(dsr PRIVATE ${ANDROID_NDK}/sources/android/native_app_glue)
    target_link_libraries(dsr PRIVATE dsrcore android log aaudio)
    # Keep ANativeActivity_onCreate exported from the glue.
    target_link_options(dsr PRIVATE -u ANativeActivity_onCreate -Wl,-z,max-page-size=16384)
else()
    add_executable(dsr_dryrun src/platform/host/dryrun_main.cpp)
    target_link_libraries(dsr_dryrun PRIVATE dsrcore pthread)
endif()
