#pragma once
// Sampling CPU profiler (diagnostics). Registered threads are sampled on their own CPU time (a SIGPROF per
// 2 ms of CPU used, so idle threads cost nothing); every 10 s the busiest places are logged per thread:
// where the CPU was (self), the first frame in our library and the first frame in the game's code.
// Addresses in our library are logged as dsr+0x<offset> for symbolization against the unstripped libdsr.so.

namespace dsr::profiler {

// Turns profiling on (settings). Threads registered before this are sampled from now on.
void Enable();
bool Enabled();
// Samples the calling thread under `name` (any thread; cheap no-op while disabled).
void RegisterCurrentThread(const char* name);

}  // namespace dsr::profiler
