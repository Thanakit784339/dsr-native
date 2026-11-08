# dsr-native — Dark Souls Remastered (Switch) native loader for Android

Runs the game's original ARM64 code (`exefs/main`) directly on an ARM64 Android device.


> [!IMPORTANT]
> **No game files are included for obvious reasons. You must obtain a legal copy of the Nintendo Switch version of Dark Souls: Remastered on your own, I cannot help you with this step.**

**NOTE: THE ASSISTANCE OF AI WAS HEAVILY USED DURING THIS PROJECT.**

## Status

**Must use v1.0.0 version of Dark Souls: Remastered for Nintendo Switch.**

Very early development. Expect many visual bugs and minor performance issues. No touch controls for now.

The only test device so far has been an AYN Thor Base model, so I cannot guarantee that this will work on any other device. I also haven't tested the v1.0.3 update for the game yet.



## How does it run?

Tested areas have been the Undead Burg, Firelink Shrine, and the Undead Asylum.

> [!IMPORTANT]
> Expect texture flickering and stuttering upon rendering in new areas. This will most likely happen the first time you load the game. Once loaded there should be no more glitching.

Runs at the Switch's default resolution of 720p.

The game is mostly playable with a very stable **60FPS** in all of my testing. Temps average around the **60c~** mark, and RAM usage is low. There are however, many minor graphical glitches that will be hopefully ironed out on later releases.

I have yet to test later areas in the game so don't expect performance to be entirely perfect.

## Working features

- [x] Fully working cutscenes and in-game rendering
- [x] Full controller support
- [x] Working lighting
- [x] Fully working audio, including NPC voice lines and sound effects
- [x] Stable 60FPS (on my test device at least)
- [x] Mailbox frame presentation for lower input latency


## Planned features
- [ ] On-screen touch controls
- [ ] Higher quality audio patch
- [ ] Further performance optimizations
- [ ] More resolutions
- [ ] Support for more phone gpus and chipsets
- [ ] User-selected game directory

## Logs

Every launch writes a log file to the phone's shared storage:

```
Download/DSRNative/dsr_YYYYMMDD_HHMMSS.txt
```

It contains device info, every runtime message, and (if the app crashes) a crash report with the
faulting address translated to `main+0x...` offsets (look these up in Ghidra with the ELF from
`tools/mkelf.py`), registers and a frame-pointer backtrace. Lines are written immediately, so the
file is complete even after a crash. The same output also goes to logcat (`adb logcat -s "DSR/*"`).






## Installing game files on the device
> [!NOTE]
> Upon first time launch a settings.txt file will also be created with a few changeable settings and their explanations.


> [!IMPORTANT]
> You must extract the games **romfs** and **exefs** files to play the game. You can use Ryujinx (Ryubing) on Windows to obtain this from your own game dump.


Install the APK and launch it once to create the necessary folders, then place the game files inside:

```
/Android/data/com.dsrnative.app/files/game/
```

If the **game** folder does not exist, you must create it.

Inside the **game** folder must be the extracted **romfs** files, along with the folders **movww**, **movjp**, and **exefs**. You must create the **exefs** folder and place your extracted **exefs** files inside.


## Known issues
- [ ] Buggy fogwalls
- [ ] NPC or player models sometimes disappearing during death animation
- [ ] Resting at a bonfire screen effect can be glitched
- [ ] Heavy effects, such as during boss fights, can cause big empty squares on screen
- [ ] Sometimes NPC's faces can appear buggy



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

On x86 it maps/relocates/binds and prints the import coverage report. On an arm64 Linux machine `--run` executes the game the same way the APK does (useful for debugging HLE without a phone).
