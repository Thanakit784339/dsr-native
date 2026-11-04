// nn::mem::StandardAllocator — a boundary-tag allocator managing the memory region the game passes to
// Initialize(), so pointers stay inside that region (engines often check ownership by address range).
// The guest object's first 8 bytes point to the host-side state.
#include <cstdint>
#include <cstring>
#include <mutex>

#include "common/log.h"
#include "hle/registry.h"

namespace {
constexpr const char* kTag = "mem";
constexpr size_t kAlign = 16;
constexpr size_t kHdr = 16;            // header: size|flags, prev_size
constexpr size_t kMinBlock = 48;       // header + free-list links + slack
constexpr int kBins = 64;
constexpr uint64_t kUsed = 1, kPrevUsed = 2, kFlags = 3;

struct Block {
    uint64_t size_flags;  // total block size incl. header, low bits = flags
    uint64_t prev_size;   // valid when the previous block is free
    // free blocks only:
    Block* next;
    Block* prev;
    size_t size() const { return size_flags & ~kFlags; }
    bool used() const { return size_flags & kUsed; }
    bool prev_used() const { return size_flags & kPrevUsed; }
};

struct Heap {
    std::mutex m;
    uint8_t* base = nullptr;
    uint8_t* end = nullptr;  // address of the sentinel header
    size_t free_bytes = 0;
    Block* bins[kBins] = {};

    static int BinOf(size_t sz) { return 63 - __builtin_clzll(sz); }
    Block* Next(Block* b) { return reinterpret_cast<Block*>(reinterpret_cast<uint8_t*>(b) + b->size()); }
    Block* Prev(Block* b) { return reinterpret_cast<Block*>(reinterpret_cast<uint8_t*>(b) - b->prev_size); }

    void Link(Block* b) {
        int i = BinOf(b->size());
        b->prev = nullptr;
        b->next = bins[i];
        if (bins[i]) bins[i]->prev = b;
        bins[i] = b;
        free_bytes += b->size();
    }
    void Unlink(Block* b) {
        int i = BinOf(b->size());
        if (b->prev) b->prev->next = b->next; else bins[i] = b->next;
        if (b->next) b->next->prev = b->prev;
        free_bytes -= b->size();
    }
    // Marks b free (with size sz), updates the following block's prev info, and links it.
    void MakeFree(Block* b, size_t sz, bool prev_used) {
        b->size_flags = sz | (prev_used ? kPrevUsed : 0);
        Block* n = Next(b);
        n->prev_size = sz;
        n->size_flags &= ~kPrevUsed;
        Link(b);
    }

    void Init(void* mem, size_t size) {
        auto a = (reinterpret_cast<uintptr_t>(mem) + kAlign - 1) & ~uintptr_t(kAlign - 1);
        size -= a - reinterpret_cast<uintptr_t>(mem);
        size &= ~(kAlign - 1);
        base = reinterpret_cast<uint8_t*>(a);
        end = base + size - kHdr;
        auto* sentinel = reinterpret_cast<Block*>(end);
        sentinel->size_flags = kUsed;  // size 0, used: stops coalescing
        auto* first = reinterpret_cast<Block*>(base);
        first->size_flags = kPrevUsed;
        MakeFree(first, size - kHdr, true);
    }

    void* Alloc(size_t n, size_t align) {
        if (align < kAlign) align = kAlign;
        size_t need = (n + kHdr + kAlign - 1) & ~(kAlign - 1);
        if (need < kMinBlock) need = kMinBlock;
        size_t search = need + (align > kAlign ? align + kMinBlock : 0);
        std::lock_guard<std::mutex> lk(m);
        Block* b = nullptr;
        for (int i = BinOf(search); i < kBins && !b; i++)
            for (Block* c = bins[i]; c; c = c->next)
                if (c->size() >= search) {
                    b = c;
                    break;
                }
        if (!b) return nullptr;
        Unlink(b);
        // Alignment: split off a free front part so the payload lands on an aligned address.
        uintptr_t payload = reinterpret_cast<uintptr_t>(b) + kHdr;
        if (payload & (align - 1)) {
            uintptr_t aligned = (payload + kMinBlock + align - 1) & ~uintptr_t(align - 1);
            size_t front = aligned - payload;
            auto* nb = reinterpret_cast<Block*>(reinterpret_cast<uint8_t*>(b) + front);
            size_t total = b->size();
            bool pu = b->prev_used();
            nb->size_flags = (total - front);
            b->size_flags = front | (pu ? kPrevUsed : 0);
            Next(b)->prev_size = front;  // Next(b) == nb now
            nb->prev_size = front;
            Link(b);
            b = nb;  // nb's prev (front part) is free: kPrevUsed stays clear
        }
        size_t total = b->size();
        if (total - need >= kMinBlock) {
            auto* rest = reinterpret_cast<Block*>(reinterpret_cast<uint8_t*>(b) + need);
            b->size_flags = need | (b->size_flags & kPrevUsed) | kUsed;
            rest->size_flags = 0;
            MakeFree(rest, total - need, true);
        } else {
            b->size_flags |= kUsed;
            Block* n = Next(b);
            n->size_flags |= kPrevUsed;
        }
        return reinterpret_cast<uint8_t*>(b) + kHdr;
    }

    void Free(void* p) {
        if (!p) return;
        std::lock_guard<std::mutex> lk(m);
        auto* b = reinterpret_cast<Block*>(static_cast<uint8_t*>(p) - kHdr);
        size_t sz = b->size();
        bool prev_used = b->prev_used();
        Block* n = Next(b);
        if (!n->used()) {
            Unlink(n);
            sz += n->size();
        }
        if (!prev_used) {
            Block* pb = Prev(b);
            Unlink(pb);
            sz += pb->size();
            b = pb;
            prev_used = pb->prev_used();
        }
        MakeFree(b, sz, prev_used);
    }

    size_t UsableSize(const void* p) {
        auto* b = reinterpret_cast<const Block*>(static_cast<const uint8_t*>(p) - kHdr);
        return b->size() - kHdr;
    }
};

Heap* H(void* self) { return *static_cast<Heap**>(self); }
}  // namespace

HLE_FUNC(_ZN2nn3mem17StandardAllocatorC1Ev, void, mem_SA_ctor, (void* self)) { *static_cast<Heap**>(self) = nullptr; }
HLE_FUNC(_ZN2nn3mem17StandardAllocator10InitializeEPvm, void, mem_SA_Initialize, (void* self, void* mem, size_t size)) {
    auto* h = new Heap;
    h->Init(mem, size);
    *static_cast<Heap**>(self) = h;
    LOGI(kTag, "StandardAllocator %p: %zu MiB at %p", self, size >> 20, mem);
}
HLE_FUNC(_ZN2nn3mem17StandardAllocator8FinalizeEv, void, mem_SA_Finalize, (void* self)) {
    delete H(self);
    *static_cast<Heap**>(self) = nullptr;
}
HLE_FUNC(_ZN2nn3mem17StandardAllocator8AllocateEm, void*, mem_SA_Allocate, (void* self, size_t n)) {
    return H(self)->Alloc(n, kAlign);
}
HLE_FUNC(_ZN2nn3mem17StandardAllocator8AllocateEmm, void*, mem_SA_AllocateAligned, (void* self, size_t n, size_t a)) {
    return H(self)->Alloc(n, a);
}
HLE_FUNC(_ZN2nn3mem17StandardAllocator4FreeEPv, void, mem_SA_Free, (void* self, void* p)) { H(self)->Free(p); }
HLE_FUNC(_ZN2nn3mem17StandardAllocator10ReallocateEPvm, void*, mem_SA_Reallocate, (void* self, void* p, size_t n)) {
    Heap* h = H(self);
    if (!p) return h->Alloc(n, kAlign);
    if (n == 0) {
        h->Free(p);
        return nullptr;
    }
    size_t old = h->UsableSize(p);
    if (n <= old) return p;
    void* q = h->Alloc(n, kAlign);
    if (!q) return nullptr;
    memcpy(q, p, old);
    h->Free(p);
    return q;
}
HLE_FUNC(_ZNK2nn3mem17StandardAllocator9GetSizeOfEPKv, size_t, mem_SA_GetSizeOf, (void* self, const void* p)) {
    return H(self)->UsableSize(p);
}
HLE_FUNC(_ZNK2nn3mem17StandardAllocator16GetTotalFreeSizeEv, size_t, mem_SA_GetTotalFreeSize, (void* self)) {
    return H(self)->free_bytes;
}
HLE_FUNC(_ZNK2nn3mem17StandardAllocator4DumpEv, void, mem_SA_Dump, (void* self)) {
    LOGI(kTag, "StandardAllocator %p: %zu bytes free", self, H(self)->free_bytes);
}
