cmake_minimum_required(VERSION 3.22)
project(dsr_native LANGUAGES C CXX ASM)

# Build identifier logged at startup: the commit CI built (GITHUB_SHA) or the local checkout.
set(DSR_BUILD_ID "$ENV{GITHUB_SHA}")
if(NOT DSR_BUILD_ID)
  execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                  OUTPUT_VARIABLE DSR_BUILD_ID OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT DSR_BUILD_ID)
  set(DSR_BUILD_ID "unknown")
endif()
string(SUBSTRING "${DSR_BUILD_ID}" 0 7 DSR_BUILD_ID)
add_compile_definitions(DSR_BUILD_ID="${DSR_BUILD_ID}")

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
target_include_directories(dsrcore PUBLIC src third_party/spirv)
target_compile_options(dsrcore PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-cast-function-type
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)
if(ANDROID)
    # The Android Gradle plugin builds its debug variant with CMAKE_BUILD_TYPE=Debug (-O0). The emulator must
    # be optimised whatever the variant: these come after the build type's flags, so they win.
    message(STATUS "dsr: CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}; native code forced to -O2")
    target_compile_options(dsrcore PRIVATE -O2 -fno-omit-frame-pointer)
endif()
if(ANDROID)
    # abi.S reads the thread_local dsr_host_x18 through TLSDESC relocations, which needs real ELF TLS
    # (API 29+). Emulated TLS would rename the symbol and break the link.
    target_compile_options(dsrcore PUBLIC -fno-emulated-tls)
endif()
if(DSR_ARM64 AND NOT ANDROID)
    # Android reserves x18 by default; on other arm64 targets reserve it explicitly so host code
    # never depends on a value guest code may have clobbered.
    target_compile_options(dsrcore PUBLIC -ffixed-x18)
endif()

if(ANDROID)
    # libadrenotools (BSD-2-Clause): loads custom Vulkan drivers (settings.txt gpu_driver) and BCeNabler. Its hook
    # libraries (libhook_impl.so, libmain_hook.so, ...) are packaged in the APK and loaded from the app's native
    # library folder (hence useLegacyPackaging in android/app/build.gradle).
    add_subdirectory(third_party/libadrenotools)
    target_include_directories(dsrcore PRIVATE third_party/libadrenotools/include)
    add_library(dsr SHARED
        src/platform/android/android_main.cpp
        ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
    target_include_directories(dsr PRIVATE ${ANDROID_NDK}/sources/android/native_app_glue)
    target_link_libraries(dsr PRIVATE dsrcore adrenotools android log aaudio mediandk z)
    # Keep ANativeActivity_onCreate exported from the glue.
    target_link_options(dsr PRIVATE -u ANativeActivity_onCreate -Wl,-z,max-page-size=16384)
else()
    add_executable(dsr_dryrun src/platform/host/dryrun_main.cpp)
    target_link_libraries(dsr_dryrun PRIVATE dsrcore pthread)
    # Translates a shader dump written by the app (tools/shader_tool/shader_tool.cpp).
    add_executable(dsr_shader_tool tools/shader_tool/shader_tool.cpp)
    target_link_libraries(dsr_shader_tool PRIVATE dsrcore pthread)
endif()

# ---- tests (host only; on arm64 they also exercise the assembly transition code) ----
if(NOT ANDROID)
    set(DSR_TEST_SOURCES tests/runtime_tests.cpp)
    if(DSR_ARM64)
        list(APPEND DSR_TEST_SOURCES tests/guest_test.S)
    endif()
    add_executable(dsr_tests ${DSR_TEST_SOURCES})
    target_link_libraries(dsr_tests PRIVATE dsrcore pthread)
    enable_testing()
    add_test(NAME dsr_tests COMMAND dsr_tests)
endif()
