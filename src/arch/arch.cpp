#include "arch/arch.h"

#include <cstdlib>

#include "common/log.h"

extern "C" {
thread_local uint64_t dsr_host_x18 = 0;
}

#if !defined(__aarch64__)
// Dry-run builds on non-arm64 hosts: guest code can be mapped and relocated but never executed.
extern "C" {
void dsr_restore_host_x18() { FATAL("arch", "guest execution is not possible on this host"); }
void dsr_unimplemented_entry() { FATAL("arch", "guest execution is not possible on this host"); }
void dsr_unimplemented_named_entry() { FATAL("arch", "guest execution is not possible on this host"); }
uint64_t dsr_guest_call(const void*, ...) { FATAL("arch", "guest execution is not possible on this host"); }
int dsr_setjmp(void*) { FATAL("arch", "guest execution is not possible on this host"); }
void dsr_longjmp(void*, int) { FATAL("arch", "guest execution is not possible on this host"); }
}
#endif

namespace dsr::arch {

void* RestoreHostX18Entry() { return reinterpret_cast<void*>(&dsr_restore_host_x18); }
void* UnimplementedEntry() { return reinterpret_cast<void*>(&dsr_unimplemented_entry); }
void* UnimplementedNamedEntry() { return reinterpret_cast<void*>(&dsr_unimplemented_named_entry); }

void FlushICache(void* start, size_t size) {
    char* s = static_cast<char*>(start);
    __builtin___clear_cache(s, s + size);
}

bool CanExecuteGuest() {
#if defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

}  // namespace dsr::arch
