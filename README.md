# dsr-native — Dark Souls Remastered (Switch) native loader for Android

Runs the game's original ARM64 code (`exefs/main`) directly on an ARM64 Android device. Nintendo's SDK
(`sdk`, `subsdk0`) is **not** executed: every one of the game's imports is bound to a host ("HLE")
implementation in this project.

```
 Android NativeActivity ──► runtime ──► loader: NSO parse → mmap → relocate → bind imports
                                          │
   game code (main, untouched ARM64) ─────┤ calls imports through thunks that restore x18
                                          ▼
                     HLE layer:  nn::os  nn::fs  nn::oe/hid/vi/audio/mem/time/...  libc  libc++  NVN
                                          ▼
                     Android:   pthreads/futex  files  AAudio  input  ANativeWindow  (Vulkan: todo)
```

## Status

| Area | State |
|---|---|
| NSO loading, relocation (171,613 relocs), 4K/16K page handling | done, verified on the real `main` |
| Import binding | **626 / 654** imported functions + all 24 data imports implemented |
| x18 handling (guest scratch vs. Android shadow call stack) | done: import thunks + `GuestCall` |
| nn::os threads, mutex, condvar, event, semaphore, rwlock, message queue, multiwait, TLS, heap | done (operates on the game's own struct layouts, taken from the SDK) |
| nn::fs → host files (`rom:` = game dir, save → app storage) | done |
| libc (musl ABI) / libc++ (`std::__1`) / operator new routing to the game's malloc | done |
| nn::mem StandardAllocator | done (boundary-tag allocator inside the game-provided region) |
| hid (gamepad), vi (window), audio out (AAudio), time, settings, account (1 local user) | done |
| Online services (nifm, socket, ssl, bcat, friends, nfp, ...) | "offline" stubs |
| **NVN graphics** | bootstrap only — every NVN call is logged by name; renderer not written yet |
| **Shaders** (Maxwell binaries in the data archives) | not started |
| movie (cutscene decoder, 28 functions) | not started (calls are logged, return 0) |

Expected first run on a device: the game boots through `nninitStartup`, its static constructors and into
`nnMain`, mounts the romfs, then stops making progress once it needs the GPU. Logcat shows the ordered
list of NVN functions it requested — that list is the spec for the Vulkan backend.

## Building

**GitHub Actions (no local tools needed)** — push this repo to GitHub. Every push runs
`.github/workflows/build.yml`, which builds the debug APK (download it from the run page under
*Artifacts* → `dsr-native-debug-apk`) and runs the runtime tests on x86-64 and arm64. You can also
start a build by hand from the *Actions* tab (*Build* → *Run workflow*).
Do not commit game files: the repo only contains source code, and `.gitignore` blocks game data.

**Android Studio** — open `android/` in Android Studio (with NDK + CMake installed) and run. Or from a shell
with the Android SDK configured: `cd android && gradle assembleDebug`.
Requirements: arm64-v8a device, Android 10 (API 29) or newer.

**Host tool** (Linux): builds the same loader and HLE layer for a dry run against the real game files:

```
cmake -S . -B build-host && cmake --build build-host -j
./build-host/dsr_dryrun /path/to/DarkSoulsR --list --verbose
```

On x86 it maps/relocates/binds and prints the import coverage report. On an arm64 Linux machine `--run`
executes the game the same way the APK does (useful for debugging HLE without a phone).

## Installing game files on the device

The app looks for the extracted game (the folder that contains `exefs/` and `dvdbnd0.bdt`) at:

```
/sdcard/Android/data/com.dsrnative.app/files/game/
```

Install the APK and launch it once (creates the folder), then:

```
adb push DarkSoulsR/. /sdcard/Android/data/com.dsrnative.app/files/game/
```

## Logs

Every launch writes a log file to the phone's shared storage:

```
Download/DSRNative/dsr_YYYYMMDD_HHMMSS.txt
```

Open it with any file manager, or over USB from a PC (Internal storage → Download → DSRNative).
It contains device info, every runtime message, and — if the app crashes — a crash report with the
faulting address translated to `main+0x...` offsets (look these up in Ghidra with the ELF from
`tools/mkelf.py`), registers and a frame-pointer backtrace. Lines are written immediately, so the
file is complete even after a crash. The same output also goes to logcat (`adb logcat -s "DSR/*"`).

## Layout

```
src/loader/     NSO parser (LZ4, SHA-256), module mapper, relocations, thunk generation
src/arch/       arm64 transition code (abi.S): x18 restore, GuestCall, musl setjmp/longjmp
src/hle/        host implementations of the game's imports (registered by mangled name)
src/hle/os/     nn::os (threads, sync, heap, ticks)
src/runtime/    load/boot sequence, guest allocator hooks, import report
src/platform/   android_main (NativeActivity), host dry-run tool
android/        Gradle project (builds src/ via CMake)
tools/          Python analysis tools (NSO → ELF with symbols for Ghidra/IDA)
```

Adding an HLE function: `HLE_FUNC(<mangled name>, ret, cxx_name, (args)) { ... }` in any `src/hle/*.cpp`.
The signature must match the guest's AAPCS64 signature (see notes in `src/hle/registry.h`).

## Design notes / known risks

* **x18**: the game uses x18 as a scratch register; Android reserves it for the shadow call stack.
  Import thunks restore the thread's host x18 (stored via TLSDESC) before entering host code;
  `GuestCall` saves/restores it around host→guest calls. Host code that calls guest function pointers
  (callbacks) must use `GuestCall`.
* **Signals**: a signal delivered while guest code runs sees the guest's x18. Fine unless a handler
  runs shadow-call-stack-instrumented code; watch for crashes inside signal handlers.
* **Memory**: the game sizes its heap to "available − used" (~3.1 GiB). The range is reserved with
  `MAP_NORESERVE`, so only touched pages consume RAM.
* **NVN struct layouts** and **Maxwell shader translation** are the large remaining sub-projects.
