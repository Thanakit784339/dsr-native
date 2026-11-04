// movie:: (multimedia.nss / subsdk0) — video playback for the intro, title and ending movies.
//
// Current backend: "no video". Extractors open any file successfully but report zero tracks, which the
// game's player handles as an empty movie; decoders refuse to start. Every call is traced so the
// player's path is visible in the log. A MediaCodec-backed implementation can replace this later; the
// .mp4 files in movww/ and movjp/ are standard H.264/AAC.
//
// movie::Extractor and movie::Decoder are polymorphic: the game constructs them in its own memory via
// the imported constructors and then calls methods through the vtable. Our constructors install a
// host vtable whose slots are x18-restoring thunks, laid out like subsdk0's vtables
// (_ZTVN5movie9ExtractorE / _ZTVN5movie7DecoderE).

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "common/log.h"
#include "hle/os/heap.h"
#include "hle/registry.h"
#include "loader/dynamic_thunks.h"
#include "runtime/runtime.h"

namespace {
constexpr const char* kTag = "movie";
constexpr int kOk = 0;
constexpr int kNotAvailable = 0x1000;  // what subsdk0's base-class methods return

// Log the first few calls of each method, then stay quiet.
void Trace(const char* what) {
    static std::mutex m;
    static std::unordered_map<const char*, int> counts;
    std::lock_guard<std::mutex> lk(m);
    int n = ++counts[what];
    if (n <= 3) LOGI(kTag, "%s", what);
}

void FreeGuestObject(void* p) {
    if (!p) return;
    if (dsr::hle::os::GuestHeapContains(p))
        dsr::GuestFree(p);
    else
        free(p);
}

// ---- Extractor vtable ---------------------------------------------------------------------------
void ExDtor(void*) { Trace("Extractor::~Extractor"); }
void ExDtorDelete(void* self) {
    Trace("Extractor::~Extractor (delete)");
    FreeGuestObject(self);
}
int ExSetDataSourcePath(void*, const char* path) {
    LOGI(kTag, "Extractor::SetDataSource(%s): videos are not supported yet, reporting an empty movie",
         path ? path : "(null)");
    return kOk;
}
int ExSetDataSourceMem(void*, const void*, int) {
    Trace("Extractor::SetDataSource(memory)");
    return kOk;
}
int ExGetTrackCount(void*, int* count) {
    Trace("Extractor::GetTrackCount -> 0");
    if (count) *count = 0;
    return kOk;
}
int ExGetTrackConfiguration(void*, int, void*) {
    Trace("Extractor::GetTrackConfiguration");
    return kNotAvailable;
}
int ExSelectTrack(void*, int) {
    Trace("Extractor::SelectTrack");
    return kNotAvailable;
}
int ExUnselectTrack(void*, int) {
    Trace("Extractor::UnselectTrack");
    return kOk;
}
int ExSeekTo(void*, int64_t) {
    Trace("Extractor::SeekTo");
    return kOk;
}
int ExGetTrackIndexForAvailableData(void*, size_t* index) {
    Trace("Extractor::GetTrackIndexForAvailableData");
    return kNotAvailable;
}
int ExRead(void*, void*) {
    Trace("Extractor::Read");
    return kNotAvailable;
}
int ExAdvance(void*) {
    Trace("Extractor::Advance");
    return kNotAvailable;
}
int ExGetSampleTime(void*, int64_t* t) {
    Trace("Extractor::GetSampleTime");
    if (t) *t = 0;
    return kNotAvailable;
}
int ExGetSampleMetaData(void*, void*) {
    Trace("Extractor::GetSampleMetaData");
    return kNotAvailable;
}
int ExGetBufferedRange(void*, int64_t* a, int64_t* b) {
    Trace("Extractor::GetBufferedRange");
    if (a) *a = 0;
    if (b) *b = 0;
    return kOk;
}
int ExSetDataSourceReader(void*, void*, const char*) {
    Trace("Extractor::SetDataSource(reader)");
    return kOk;
}

// ---- Decoder vtable -------------------------------------------------------------------------------
void DecDtor(void*) { Trace("Decoder::~Decoder"); }
void DecDtorDelete(void* self) {
    Trace("Decoder::~Decoder (delete)");
    FreeGuestObject(self);
}
int DecConfigure(void*, void*) {
    Trace("Decoder::Configure");
    return kNotAvailable;
}
int DecRegisterClientEvents(void*, void*) {
    Trace("Decoder::RegisterClientEvents");
    return kOk;
}
int DecGetOutputFormat(void*, void*) {
    Trace("Decoder::GetOutputFormat");
    return kNotAvailable;
}
int DecStart(void*) {
    Trace("Decoder::Start");
    return kNotAvailable;
}
int DecStop(void*) {
    Trace("Decoder::Stop");
    return kOk;
}
int DecReset(void*) { return kOk; }
int DecFlush(void*) { return kOk; }
int DecAcquireInputBufferIndex(void*, int* index, int*) {
    Trace("Decoder::AcquireInputBufferIndex");
    if (index) *index = -1;
    return kNotAvailable;
}
int DecGetInputBuffer(void*, int, void*) {
    Trace("Decoder::GetInputBuffer");
    return kNotAvailable;
}
int DecSendInputBufferForDecode(void*, int, int, int, int64_t, uint32_t) {
    Trace("Decoder::SendInputBufferForDecode");
    return kNotAvailable;
}
int DecAcquireOutputBufferIndex(void*, int* index, int64_t*, uint32_t*, int*) {
    Trace("Decoder::AcquireOutputBufferIndex");
    if (index) *index = -1;
    return kNotAvailable;
}
int DecGetOutputBuffer(void*, int, void*) {
    Trace("Decoder::GetOutputBuffer");
    return kNotAvailable;
}
int DecGetOutputBufferSize(void*, int* a, int* b) {
    if (a) *a = 0;
    if (b) *b = 0;
    return kNotAvailable;
}
int DecReleaseOutputBufferIndex(void*, int) { return kOk; }
int DecGetLastError(void*) { return kNotAvailable; }
int DecGetOutputBuffer3(void*, int, void*, void*) { return kNotAvailable; }
int DecGetOutputBufferFormat(void*, int, void*) { return kNotAvailable; }

// Builds {offset-to-top, typeinfo, slots...}; returns the address an object's vptr must hold.
template <size_t N>
const void* BuildVtable(void* const (&fns)[N]) {
    auto* vt = new uint64_t[N + 2]();
    for (size_t i = 0; i < N; i++) vt[i + 2] = reinterpret_cast<uint64_t>(dsr::loader::MakeDynamicThunk(fns[i], 0));
    return vt + 2;
}
#define F(f) reinterpret_cast<void*>(&f)

const void* ExtractorVptr() {
    static void* const fns[] = {F(ExDtor), F(ExDtorDelete), F(ExSetDataSourcePath), F(ExSetDataSourceMem),
                                F(ExGetTrackCount), F(ExGetTrackConfiguration), F(ExSelectTrack),
                                F(ExUnselectTrack), F(ExSeekTo), F(ExGetTrackIndexForAvailableData), F(ExRead),
                                F(ExAdvance), F(ExGetSampleTime), F(ExGetSampleMetaData), F(ExGetBufferedRange),
                                F(ExSetDataSourceReader)};
    static const void* vptr = BuildVtable(fns);
    return vptr;
}

const void* DecoderVptr() {
    static void* const fns[] = {F(DecDtor), F(DecDtorDelete), F(DecConfigure), F(DecRegisterClientEvents),
                                F(DecGetOutputFormat), F(DecStart), F(DecStop), F(DecReset), F(DecFlush),
                                F(DecAcquireInputBufferIndex), F(DecGetInputBuffer), F(DecSendInputBufferForDecode),
                                F(DecAcquireOutputBufferIndex), F(DecGetOutputBuffer), F(DecGetOutputBufferSize),
                                F(DecReleaseOutputBufferIndex), F(DecGetLastError), F(DecGetOutputBuffer3),
                                F(DecGetOutputBufferFormat)};
    static const void* vptr = BuildVtable(fns);
    return vptr;
}
#undef F

// ---- Buffer: state kept host-side (object size is private to subsdk0) ------------------------------
struct BufferState {
    uint8_t* data = nullptr;
    size_t capacity = 0;
    int offset = 0;
    int size = 0;
    bool owned = false;
};
std::mutex g_buf_mutex;
std::unordered_map<const void*, BufferState> g_buffers;

BufferState& Buf(const void* self) {
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    return g_buffers[self];
}
}  // namespace

// ===== global configuration ======================================================================
HLE_FUNC(_ZN5movie12SetAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_, void, movie_SetAllocator,
         (const void*, const void*, const void*, void*)) {}
HLE_FUNC(_ZN5movie11SetCoreMaskEm, void, movie_SetCoreMask, (uint64_t)) {}

// ===== Extractor / Decoder construction ==============================================================
HLE_FUNC(_ZN5movie9ExtractorC1ENS_13ContainerTypeE, void, movie_ExtractorCtor, (void* self, int container)) {
    Trace("Extractor::Extractor");
    memset(self, 0, 16);
    *static_cast<const void**>(self) = ExtractorVptr();
}
HLE_FUNC(_ZN5movie7DecoderC1ENS_11DecoderTypeENS_19DecoderOutputFormatENS_11DecoderModeE, void, movie_DecoderCtor,
         (void* self, int type, int format, int mode)) {
    Trace("Decoder::Decoder");
    *static_cast<const void**>(self) = DecoderVptr();
}
HLE_FUNC(_ZN5movie7DecoderD1Ev, void, movie_DecoderDtor, (void* self)) { DecDtor(self); }
HLE_FUNC(_ZN5movie7Decoder9ConfigureEPNS_9MediaDataE, int, movie_DecoderConfigure, (void* s, void* m)) {
    return DecConfigure(s, m);
}
HLE_FUNC(_ZN5movie7Decoder20RegisterClientEventsEPNS_13DecoderEventsE, int, movie_DecoderRegister, (void* s, void* e)) {
    return DecRegisterClientEvents(s, e);
}
HLE_FUNC(_ZN5movie7Decoder5StartEv, int, movie_DecoderStart, (void* s)) { return DecStart(s); }
HLE_FUNC(_ZN5movie7Decoder4StopEv, int, movie_DecoderStop, (void* s)) { return DecStop(s); }
HLE_FUNC(_ZN5movie7Decoder23AcquireInputBufferIndexEPiS1_, int, movie_DecoderAcqIn, (void* s, int* a, int* b)) {
    return DecAcquireInputBufferIndex(s, a, b);
}
HLE_FUNC(_ZN5movie7Decoder14GetInputBufferEiPNS_6BufferE, int, movie_DecoderGetIn, (void* s, int i, void* b)) {
    return DecGetInputBuffer(s, i, b);
}
HLE_FUNC(_ZN5movie7Decoder24SendInputBufferForDecodeEiiilj, int, movie_DecoderSend,
         (void* s, int i, int o, int n, int64_t t, uint32_t f)) {
    return DecSendInputBufferForDecode(s, i, o, n, t, f);
}
HLE_FUNC(_ZN5movie7Decoder24AcquireOutputBufferIndexEPiPlPjS1_, int, movie_DecoderAcqOut,
         (void* s, int* i, int64_t* t, uint32_t* f, int* c)) {
    return DecAcquireOutputBufferIndex(s, i, t, f, c);
}
HLE_FUNC(_ZN5movie7Decoder15GetOutputBufferEiPNS_6BufferE, int, movie_DecoderGetOut, (void* s, int i, void* b)) {
    return DecGetOutputBuffer(s, i, b);
}
HLE_FUNC(_ZN5movie7Decoder24ReleaseOutputBufferIndexEi, int, movie_DecoderRelease, (void* s, int i)) {
    return DecReleaseOutputBufferIndex(s, i);
}

// ===== MediaData (no metadata available) ==============================================================
HLE_FUNC(_ZN5movie9MediaDataC1Ev, void, movie_MediaDataCtor, (void*)) {}
HLE_FUNC(_ZN5movie9MediaDataC1ERKS0_, void, movie_MediaDataCopy, (void*, const void*)) {}
HLE_FUNC(_ZN5movie9MediaDataD1Ev, void, movie_MediaDataDtor, (void*)) {}
HLE_FUNC(_ZNK5movie9MediaData10FindStringEPKcPS2_, bool, movie_MediaDataFindString,
         (const void*, const char* key, const char** out)) {
    if (out) *out = nullptr;
    return false;
}
HLE_FUNC(_ZNK5movie9MediaData9FindInt32EPKcPi, bool, movie_MediaDataFindInt32, (const void*, const char*, int* out)) {
    if (out) *out = 0;
    return false;
}

// ===== Buffer ===========================================================================================
HLE_FUNC(_ZN5movie6BufferC1Ev, void, movie_BufferCtor, (void* self)) {
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    g_buffers[self] = BufferState{};
}
HLE_FUNC(_ZN5movie6BufferC1Em, void, movie_BufferCtorSize, (void* self, size_t n)) {
    BufferState s;
    s.data = static_cast<uint8_t*>(calloc(1, n ? n : 1));
    s.capacity = n;
    s.owned = true;
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    g_buffers[self] = s;
}
HLE_FUNC(_ZN5movie6BufferD1Ev, void, movie_BufferDtor, (void* self)) {
    std::lock_guard<std::mutex> lk(g_buf_mutex);
    auto it = g_buffers.find(self);
    if (it == g_buffers.end()) return;
    if (it->second.owned) free(it->second.data);
    g_buffers.erase(it);
}
HLE_FUNC(_ZN5movie6Buffer18SetDataAndCapacityEPvm, void, movie_BufferSetData, (void* self, void* data, size_t cap)) {
    BufferState& s = Buf(self);
    if (s.owned) free(s.data);
    s.data = static_cast<uint8_t*>(data);
    s.capacity = cap;
    s.owned = false;
    s.offset = 0;
    s.size = 0;
}
HLE_FUNC(_ZN5movie6Buffer8SetRangeEii, void, movie_BufferSetRange, (void* self, int offset, int size)) {
    BufferState& s = Buf(self);
    s.offset = offset;
    s.size = size;
}
HLE_FUNC(_ZNK5movie6Buffer4BaseEv, void*, movie_BufferBase, (const void* self)) { return Buf(self).data; }
HLE_FUNC(_ZNK5movie6Buffer4SizeEv, size_t, movie_BufferSize, (const void* self)) { return size_t(Buf(self).size); }
HLE_FUNC(_ZNK5movie6Buffer6OffsetEv, size_t, movie_BufferOffset, (const void* self)) {
    return size_t(Buf(self).offset);
}
