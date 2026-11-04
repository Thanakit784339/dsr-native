// Maxwell -> SPIR-V translator. See translator.h for scope and the resource interface.
//
// Model: each Maxwell GPR is a Function-storage uint variable (the driver's compiler promotes them to
// SSA); predicates are bool variables. Instructions are translated one at a time in program order; a
// guarded instruction (@P0 / @!P0) becomes a structured if around its body. Values that matter for
// static resource decisions (texture handles loaded from the NVN texture-binding table c[13], constant
// texel offsets set with MOV32I) are tracked through registers while translating.
//
// Field layouts follow the publicly documented Maxwell ISA encodings (envytools / open-source emulator
// research); semantics notes are inline.

#include "hle/nvn/shader/translator.h"

#include <algorithm>
#include <cstring>
#include <map>

#include "hle/nvn/shader/spirv_builder.h"

namespace dsr::hle::nvn::shader {
namespace {

using spv::Id;
using G = spv::Builder::GLSLstd450Shim;

constexpr size_t kCodeOffset = 0x80;     // instructions start after the NVN header + SPH
constexpr size_t kSphOffset = 0x30;
constexpr uint32_t kTextureTableCbuf = 13;
constexpr uint32_t kTextureTableBase = 0x20;
constexpr uint32_t kMaxCbufs = 18;
constexpr uint32_t kCbufVec4s = 4096;    // 64 KiB view per constant buffer

inline uint64_t Bits(uint64_t v, int lo, int n) { return (v >> lo) & ((n >= 64) ? ~0ull : ((1ull << n) - 1)); }

// ---- opcode classification (most specific pattern first) ---------------------------------------------------
enum class Op {
    Unknown, AL2P, ALD, AST, BRA, BRK, CONT, DEPBAR, EXIT, F2F, F2I, FADD, FADD32I, FFMA, FFMA32I, FMNMX, FMUL,
    FMUL32I, FSET, FSETP, I2F, I2I, IADD, IADD3, IADD32I, IPA, ISCADD, ISET, ISETP, KIL, LDC, LDL, LOP, LOP32I,
    MOV, MOV32I, MUFU, NOP, PBK, PCNT, PSETP, RRO, SEL, SHL, SHR, SSY, STL, SULD, SUST, SYNC, TEXB, TEX, TEXS, TLD,
    TLDB, TLD4, TLD4B, TLDS, XMAD, MEMBAR, BAR, IMNMX, LOP3, ICMP, FCMP, PRMT, SHF, POPC, FLO, BFE, BFI, IMUL,
    IMAD, LEA, HADD2, HMUL2, HFMA2, R2P, P2R, S2R, CS2R, OUT, SHFL, VOTE, ATOM, RED, LDG, STG, LDS, STS, ST, LD,
};
enum class Form { None, Reg, Cbuf, Imm, RC, CR };

struct Pattern {
    const char* bits;
    Op op;
    Form form;
};
// 16 most significant bits; '-' = don't care.
const Pattern kPatterns[] = {
    {"1110111111011---", Op::ALD, Form::None},     {"1110111111110---", Op::AST, Form::None},
    {"1110111110100---", Op::AL2P, Form::None},    {"111000100100----", Op::BRA, Form::None},
    {"111000110100----", Op::BRK, Form::None},     {"111000110101----", Op::CONT, Form::None},
    {"1111000011110---", Op::DEPBAR, Form::None},  {"111000110000----", Op::EXIT, Form::None},
    {"0101110010101---", Op::F2F, Form::Reg},      {"0100110010101---", Op::F2F, Form::Cbuf},
    {"0011100-10101---", Op::F2F, Form::Imm},      {"0101110010110---", Op::F2I, Form::Reg},
    {"0100110010110---", Op::F2I, Form::Cbuf},     {"0011100-10110---", Op::F2I, Form::Imm},
    {"0101110001011---", Op::FADD, Form::Reg},     {"0100110001011---", Op::FADD, Form::Cbuf},
    {"0011100-01011---", Op::FADD, Form::Imm},     {"000010----------", Op::FADD32I, Form::None},
    {"010110011-------", Op::FFMA, Form::Reg},     {"010100011-------", Op::FFMA, Form::RC},
    {"010010011-------", Op::FFMA, Form::CR},      {"0011001-1-------", Op::FFMA, Form::Imm},
    {"000011----------", Op::FFMA32I, Form::None}, {"0101110001100---", Op::FMNMX, Form::Reg},
    {"0100110001100---", Op::FMNMX, Form::Cbuf},   {"0011100-01100---", Op::FMNMX, Form::Imm},
    {"0101110001101---", Op::FMUL, Form::Reg},     {"0100110001101---", Op::FMUL, Form::Cbuf},
    {"0011100-01101---", Op::FMUL, Form::Imm},     {"00011110--------", Op::FMUL32I, Form::None},
    {"01011000--------", Op::FSET, Form::Reg},     {"01001000--------", Op::FSET, Form::Cbuf},
    {"0011000---------", Op::FSET, Form::Imm},     {"010110111011----", Op::FSETP, Form::Reg},
    {"010010111011----", Op::FSETP, Form::Cbuf},   {"0011011-1011----", Op::FSETP, Form::Imm},
    {"0101110010111---", Op::I2F, Form::Reg},      {"0100110010111---", Op::I2F, Form::Cbuf},
    {"0011100-10111---", Op::I2F, Form::Imm},      {"0101110011100---", Op::I2I, Form::Reg},
    {"0100110011100---", Op::I2I, Form::Cbuf},     {"0011100-11100---", Op::I2I, Form::Imm},
    {"0101110000010---", Op::IADD, Form::Reg},     {"0100110000010---", Op::IADD, Form::Cbuf},
    {"0011100-00010---", Op::IADD, Form::Imm},     {"010111001100----", Op::IADD3, Form::Reg},
    {"010011001100----", Op::IADD3, Form::Cbuf},   {"0011100-1100----", Op::IADD3, Form::Imm},
    {"0001110---------", Op::IADD32I, Form::None}, {"11100000--------", Op::IPA, Form::None},
    {"0101110000011---", Op::ISCADD, Form::Reg},   {"0100110000011---", Op::ISCADD, Form::Cbuf},
    {"0011100-00011---", Op::ISCADD, Form::Imm},   {"010110110101----", Op::ISET, Form::Reg},
    {"010010110101----", Op::ISET, Form::Cbuf},    {"0011011-0101----", Op::ISET, Form::Imm},
    {"010110110110----", Op::ISETP, Form::Reg},    {"010010110110----", Op::ISETP, Form::Cbuf},
    {"0011011-0110----", Op::ISETP, Form::Imm},    {"111000110011----", Op::KIL, Form::None},
    {"1110111110010---", Op::LDC, Form::None},     {"1110111101000---", Op::LDL, Form::None},
    {"0101110001000---", Op::LOP, Form::Reg},      {"0100110001000---", Op::LOP, Form::Cbuf},
    {"0011100-01000---", Op::LOP, Form::Imm},      {"000001----------", Op::LOP32I, Form::None},
    {"0101110010011---", Op::MOV, Form::Reg},      {"0100110010011---", Op::MOV, Form::Cbuf},
    {"0011100-10011---", Op::MOV, Form::Imm},      {"000000010000----", Op::MOV32I, Form::None},
    {"0101000010000---", Op::MUFU, Form::None},    {"0101000010110---", Op::NOP, Form::None},
    {"111000101010----", Op::PBK, Form::None},     {"111000101011----", Op::PCNT, Form::None},
    {"0101000010010---", Op::PSETP, Form::None},   {"0101110010010---", Op::RRO, Form::Reg},
    {"0100110010010---", Op::RRO, Form::Cbuf},     {"0011100-10010---", Op::RRO, Form::Imm},
    {"0101110010100---", Op::SEL, Form::Reg},      {"0100110010100---", Op::SEL, Form::Cbuf},
    {"0011100-10100---", Op::SEL, Form::Imm},      {"0101110001001---", Op::SHL, Form::Reg},
    {"0100110001001---", Op::SHL, Form::Cbuf},     {"0011100-01001---", Op::SHL, Form::Imm},
    {"0101110000101---", Op::SHR, Form::Reg},      {"0100110000101---", Op::SHR, Form::Cbuf},
    {"0011100-00101---", Op::SHR, Form::Imm},      {"111000101001----", Op::SSY, Form::None},
    {"1110111101010---", Op::STL, Form::None},     {"111010110010----", Op::SUST, Form::None}, {"1110101100------", Op::SULD, Form::None},
    {"1111000011111---", Op::SYNC, Form::None},    {"1101111010------", Op::TEXB, Form::None},
    {"11000-----------", Op::TEX, Form::None},     {"1101-00---------", Op::TEXS, Form::None},
    {"11011100--------", Op::TLD, Form::None},     {"11011101--------", Op::TLDB, Form::None},
    {"110010----------", Op::TLD4, Form::None},    {"1101111011------", Op::TLD4B, Form::None},
    {"1101-01---------", Op::TLDS, Form::None},    {"0101101100------", Op::XMAD, Form::Reg},
    {"010100010-------", Op::XMAD, Form::RC},      {"01001110--------", Op::XMAD, Form::CR},
    {"0011011-00------", Op::XMAD, Form::Imm},     {"1110111110011---", Op::MEMBAR, Form::None},
    {"1111000010101---", Op::BAR, Form::None},
};

struct Decoded {
    Op op = Op::Unknown;
    Form form = Form::None;
};

struct PatternTable {
    struct Entry {
        uint16_t mask, value;
        int specificity;
        Op op;
        Form form;
    };
    std::vector<Entry> entries;
    PatternTable() {
        for (const Pattern& p : kPatterns) {
            Entry e{0, 0, 0, p.op, p.form};
            for (int i = 0; i < 16; i++) {
                uint16_t bit = uint16_t(1u << (15 - i));
                if (p.bits[i] != '-') {
                    e.mask |= bit;
                    e.specificity++;
                    if (p.bits[i] == '1') e.value |= bit;
                }
            }
            entries.push_back(e);
        }
        std::stable_sort(entries.begin(), entries.end(),
                         [](const Entry& a, const Entry& b) { return a.specificity > b.specificity; });
    }
    Decoded Decode(uint64_t inst) const {
        uint16_t top = uint16_t(inst >> 48);
        for (const Entry& e : entries)
            if ((top & e.mask) == e.value) return {e.op, e.form};
        return {};
    }
};
const PatternTable& Patterns() {
    static const PatternTable t;
    return t;
}

const char* OpName(Op op) {
    switch (op) {
#define N(x) case Op::x: return #x;
        N(AL2P) N(ALD) N(AST) N(BRA) N(BRK) N(CONT) N(DEPBAR) N(EXIT) N(F2F) N(F2I) N(FADD) N(FADD32I) N(FFMA)
        N(FFMA32I) N(FMNMX) N(FMUL) N(FMUL32I) N(FSET) N(FSETP) N(I2F) N(I2I) N(IADD) N(IADD3) N(IADD32I) N(IPA)
        N(ISCADD) N(ISET) N(ISETP) N(KIL) N(LDC) N(LDL) N(LOP) N(LOP32I) N(MOV) N(MOV32I) N(MUFU) N(NOP) N(PBK)
        N(PCNT) N(PSETP) N(RRO) N(SEL) N(SHL) N(SHR) N(SSY) N(STL) N(SULD) N(SUST) N(SYNC) N(TEXB) N(TEX) N(TEXS) N(TLD)
        N(TLDB) N(TLD4) N(TLD4B) N(TLDS) N(XMAD) N(MEMBAR) N(BAR)
#undef N
        default: return "unknown";
    }
}

// ---- translation context ----------------------------------------------------------------------------------
struct Known {
    enum Kind : uint8_t { None, Const, TexSlot } kind = None;
    uint32_t value = 0;
};

class Translator {
public:
    Translator(const uint8_t* code, size_t size, const TranslateOptions& opt) : code_(code), size_(size), opt_(opt) {}

    TranslateResult Run() {
        TranslateResult r;
        if (size_ < kCodeOffset + 8) return Fail(r, "shader too small");
        uint32_t sph0;
        memcpy(&sph0, code_ + kSphOffset, 4);
        uint32_t type = (sph0 >> 10) & 0xF;
        if (type == 1)
            stage_ = Stage::Vertex;
        else if (type == 5)
            stage_ = Stage::Fragment;
        else
            return Fail(r, "unsupported shader stage " + std::to_string(type));
        info_.stage = stage_;
        Setup();
        const size_t count = (size_ - kCodeOffset) / 8;
        bool ended = false;
        for (size_t idx = 0; idx < count && !ended && error_.empty(); idx++) {
            if (idx % 4 == 0) continue;  // scheduling control word
            uint64_t inst;
            memcpy(&inst, code_ + kCodeOffset + idx * 8, 8);
            pc_ = uint32_t(idx * 8);
            // Close conditional blocks that end here (their SSY reconvergence point).
            while (!blocks_.empty() && blocks_.back().end == pc_) {
                if (!blocks_.back().closed) b_.Op(SpvOpBranch, {blocks_.back().merge});
                b_.Label(blocks_.back().merge);
                // After the merge a register is only known if the block may not have changed it.
                const auto& before = blocks_.back().known_before;
                for (int r = 0; r < 256; r++)
                    if (known_[r].kind != before[r].kind || known_[r].value != before[r].value) known_[r] = Known{};
                blocks_.pop_back();
            }
            while (!ssy_.empty() && ssy_.back() == pc_) ssy_.pop_back();
            if (skip_until_) {
                if (pc_ < skip_until_) continue;  // jumped over (dead code)
                skip_until_ = 0;
            }
            info_.instruction_count++;
            ended = Emit(inst);
        }
        if (!error_.empty()) return Fail(r, error_);
        if (!ended) return Fail(r, "no EXIT found");
        // Entry point + execution modes.
        SpvExecutionModel model = stage_ == Stage::Vertex ? SpvExecutionModelVertex : SpvExecutionModelFragment;
        b_.EntryPoint(model, main_, "main", interface_);
        if (stage_ == Stage::Fragment) {
            b_.ExecutionMode(main_, SpvExecutionModeOriginUpperLeft);
            if (info_.writes_depth) b_.ExecutionMode(main_, SpvExecutionModeDepthReplacing);
        }
        r.spirv = b_.Finish(main_, first_label_);
        r.info = info_;
        r.ok = true;
        return r;
    }

private:
    TranslateResult& Fail(TranslateResult& r, const std::string& why) {
        r.ok = false;
        r.error = why;
        return r;
    }
    void Error(const std::string& why) {
        if (error_.empty()) {
            char buf[32];
            snprintf(buf, sizeof(buf), " (at 0x%x)", pc_);
            error_ = why + buf;
        }
    }

    // ---- setup / types ----------------------------------------------------------------------------------------
    void Setup() {
        b_.Capability(SpvCapabilityShader);
        main_ = b_.NewId();
        first_label_ = b_.NewId();
        t_void_ = b_.TypeVoid();
        t_bool_ = b_.TypeBool();
        t_f32_ = b_.TypeFloat();
        t_u32_ = b_.TypeUint();
        t_i32_ = b_.TypeInt();
        t_v4f_ = b_.TypeVector(t_f32_, 4);
        t_v4u_ = b_.TypeVector(t_u32_, 4);
        t_v4i_ = b_.TypeVector(t_i32_, 4);
        t_v2f_ = b_.TypeVector(t_f32_, 2);
        t_v3f_ = b_.TypeVector(t_f32_, 3);
        t_v2i_ = b_.TypeVector(t_i32_, 2);
        t_v3i_ = b_.TypeVector(t_i32_, 3);
        p_fn_u32_ = b_.TypePointer(SpvStorageClassFunction, t_u32_);
        p_fn_bool_ = b_.TypePointer(SpvStorageClassFunction, t_bool_);
        b_.Name(main_, "main");
    }
    Id DescriptorSet() const { return stage_ == Stage::Vertex ? 0 : 1; }

    // ---- registers --------------------------------------------------------------------------------------------
    Id RegVar(uint32_t r) {
        if (!reg_[r]) {
            reg_[r] = b_.LocalVariable(p_fn_u32_);
            char name[8];
            snprintf(name, sizeof(name), "R%u", r);
            b_.Name(reg_[r], name);
        }
        return reg_[r];
    }
    Id GetU(uint32_t r) {
        if (r == 255) return b_.ConstUint(0);
        return b_.OpResult(SpvOpLoad, t_u32_, {RegVar(r)});
    }
    Id GetF(uint32_t r) {
        if (r == 255) return b_.ConstFloat(0.0f);
        return BitcastF(GetU(r));
    }
    void SetU(uint32_t r, Id v, Known k = {}) {
        if (r == 255) return;
        b_.Op(SpvOpStore, {RegVar(r), v});
        known_[r] = predicated_ ? Known{} : k;
    }
    void SetF(uint32_t r, Id f) { SetU(r, BitcastU(f)); }
    Id BitcastF(Id u) { return b_.OpResult(SpvOpBitcast, t_f32_, {u}); }
    Id BitcastU(Id v) { return b_.OpResult(SpvOpBitcast, t_u32_, {v}); }
    Id BitcastI(Id v) { return b_.OpResult(SpvOpBitcast, t_i32_, {v}); }

    Id PredVar(uint32_t p) {
        if (!pred_[p]) {
            pred_[p] = b_.LocalVariable(p_fn_bool_, b_.ConstBool(false));  // predicates start false
            char name[8];
            snprintf(name, sizeof(name), "P%u", p);
            b_.Name(pred_[p], name);
        }
        return pred_[p];
    }
    Id GetPred(uint32_t p, bool neg) {
        Id v = (p == 7) ? b_.ConstBool(true) : b_.OpResult(SpvOpLoad, t_bool_, {PredVar(p)});
        return neg ? b_.OpResult(SpvOpLogicalNot, t_bool_, {v}) : v;
    }
    void SetPred(uint32_t p, Id v) {
        if (p == 7) return;
        b_.Op(SpvOpStore, {PredVar(p), v});
    }

    // ---- constant buffers ----------------------------------------------------------------------------------
    Id CbufVar(uint32_t idx) {
        if (idx >= kMaxCbufs) {
            Error("constant buffer index out of range");
            idx = 0;
        }
        if (!cbuf_[idx]) {
            Id arr = b_.TypeArray(t_v4u_, kCbufVec4s);
            b_.Decorate(arr, SpvDecorationArrayStride, {16});
            Id st = b_.TypeStruct({arr});
            b_.Decorate(st, SpvDecorationBlock);
            b_.MemberDecorate(st, 0, SpvDecorationOffset, {0});
            cbuf_[idx] = b_.Variable(b_.TypePointer(SpvStorageClassUniform, st), SpvStorageClassUniform);
            b_.Decorate(cbuf_[idx], SpvDecorationDescriptorSet, {DescriptorSet()});
            b_.Decorate(cbuf_[idx], SpvDecorationBinding, {idx});
            char name[16];
            snprintf(name, sizeof(name), "c%u", idx);
            b_.Name(cbuf_[idx], name);
            info_.cbuf_mask |= 1u << idx;
        }
        return cbuf_[idx];
    }
    void CbufReach(uint32_t idx, uint32_t bytes) {
        if (idx < 18) info_.cbuf_bytes[idx] = std::max(info_.cbuf_bytes[idx], std::min<uint32_t>(bytes, kCbufVec4s * 16));
    }
    Id Cbuf(uint32_t idx, uint32_t byte_offset) {
        Id var = CbufVar(idx);
        CbufReach(idx, ((byte_offset % (kCbufVec4s * 16)) | 15) + 1);
        Id p = b_.OpResult(SpvOpAccessChain, b_.TypePointer(SpvStorageClassUniform, t_u32_),
                           {var, b_.ConstUint(0), b_.ConstUint((byte_offset >> 4) % kCbufVec4s),
                            b_.ConstUint((byte_offset >> 2) & 3)});
        return b_.OpResult(SpvOpLoad, t_u32_, {p});
    }
    Id CbufDynamic(uint32_t idx, Id byte_offset) {
        Id var = CbufVar(idx);
        CbufReach(idx, kCbufVec4s * 16);
        Id vec_index = b_.OpResult(SpvOpShiftRightLogical, t_u32_, {byte_offset, b_.ConstUint(4)});
        vec_index = b_.OpResult(SpvOpBitwiseAnd, t_u32_, {vec_index, b_.ConstUint(kCbufVec4s - 1)});
        Id comp = b_.OpResult(SpvOpShiftRightLogical, t_u32_, {byte_offset, b_.ConstUint(2)});
        comp = b_.OpResult(SpvOpBitwiseAnd, t_u32_, {comp, b_.ConstUint(3)});
        Id p = b_.OpResult(SpvOpAccessChain, b_.TypePointer(SpvStorageClassUniform, t_v4u_),
                           {var, b_.ConstUint(0), vec_index});
        Id v = b_.OpResult(SpvOpLoad, t_v4u_, {p});
        return b_.OpResult(SpvOpVectorExtractDynamic, t_u32_, {v, comp});
    }

    // ---- operands ---------------------------------------------------------------------------------------------
    static uint32_t Imm20Float(uint64_t i) { return uint32_t((Bits(i, 20, 19) | (Bits(i, 56, 1) << 19)) << 12); }
    static uint32_t Imm20Int(uint64_t i) {
        uint32_t v = uint32_t(Bits(i, 20, 19) | (Bits(i, 56, 1) << 19));
        return (v & 0x80000) ? (v | 0xFFF00000u) : v;
    }
    Id SrcBU(uint64_t i, Form f, bool is_float) {
        switch (f) {
            case Form::Reg: return GetU(uint32_t(Bits(i, 20, 8)));
            case Form::Cbuf:
            case Form::CR: return Cbuf(uint32_t(Bits(i, 34, 5)), uint32_t(Bits(i, 20, 14) * 4));
            case Form::RC: return GetU(uint32_t(Bits(i, 39, 8)));
            case Form::Imm: return b_.ConstUint(is_float ? Imm20Float(i) : Imm20Int(i));
            default: Error("bad operand form"); return b_.ConstUint(0);
        }
    }
    Id SrcBF(uint64_t i, Form f) { return BitcastF(SrcBU(i, f, true)); }
    Id SrcCU(uint64_t i, Form f) {
        if (f == Form::RC) return Cbuf(uint32_t(Bits(i, 34, 5)), uint32_t(Bits(i, 20, 14) * 4));
        return GetU(uint32_t(Bits(i, 39, 8)));
    }
    Id SrcCF(uint64_t i, Form f) { return BitcastF(SrcCU(i, f)); }
    // Constant value of operand B if statically known (used for texel offsets).
    bool KnownB(uint64_t i, Form f, uint32_t* out) {
        if (f == Form::Imm) {
            *out = Imm20Int(i);
            return true;
        }
        if (f == Form::Reg) {
            const Known& k = known_[Bits(i, 20, 8)];
            if (k.kind == Known::Const) {
                *out = k.value;
                return true;
            }
        }
        return false;
    }

    Id FNeg(Id v) { return b_.OpResult(SpvOpFNegate, t_f32_, {v}); }
    Id FAbs(Id v) { return b_.Glsl(t_f32_, G::kFAbs, {v}); }
    Id FMod(Id v, bool abs, bool neg) {
        if (abs) v = FAbs(v);
        if (neg) v = FNeg(v);
        return v;
    }
    Id Sat(Id v, bool sat) {
        return sat ? b_.Glsl(t_f32_, G::kFClamp, {v, b_.ConstFloat(0.0f), b_.ConstFloat(1.0f)}) : v;
    }

    Id FCompare(uint32_t op, Id a, Id bv) {
        SpvOp o;
        switch (op) {
            case 0: return b_.ConstBool(false);
            case 1: o = SpvOpFOrdLessThan; break;
            case 2: o = SpvOpFOrdEqual; break;
            case 3: o = SpvOpFOrdLessThanEqual; break;
            case 4: o = SpvOpFOrdGreaterThan; break;
            case 5: o = SpvOpFOrdNotEqual; break;
            case 6: o = SpvOpFOrdGreaterThanEqual; break;
            case 7:
            case 8: {
                Id na = b_.OpResult(SpvOpIsNan, t_bool_, {a});
                Id nb = b_.OpResult(SpvOpIsNan, t_bool_, {bv});
                Id any = b_.OpResult(SpvOpLogicalOr, t_bool_, {na, nb});
                return op == 8 ? any : b_.OpResult(SpvOpLogicalNot, t_bool_, {any});
            }
            case 9: o = SpvOpFUnordLessThan; break;
            case 10: o = SpvOpFUnordEqual; break;
            case 11: o = SpvOpFUnordLessThanEqual; break;
            case 12: o = SpvOpFUnordGreaterThan; break;
            case 13: o = SpvOpFUnordNotEqual; break;
            case 14: o = SpvOpFUnordGreaterThanEqual; break;
            default: return b_.ConstBool(true);
        }
        return b_.OpResult(o, t_bool_, {a, bv});
    }
    Id ICompare(uint32_t op, bool is_signed, Id a, Id bv) {
        SpvOp o;
        switch (op) {
            case 0: return b_.ConstBool(false);
            case 1: o = is_signed ? SpvOpSLessThan : SpvOpULessThan; break;
            case 2: o = SpvOpIEqual; break;
            case 3: o = is_signed ? SpvOpSLessThanEqual : SpvOpULessThanEqual; break;
            case 4: o = is_signed ? SpvOpSGreaterThan : SpvOpUGreaterThan; break;
            case 5: o = SpvOpINotEqual; break;
            case 6: o = is_signed ? SpvOpSGreaterThanEqual : SpvOpUGreaterThanEqual; break;
            default: return b_.ConstBool(true);
        }
        return b_.OpResult(o, t_bool_, {a, bv});
    }
    Id BoolOp(uint32_t op, Id a, Id bv) {
        switch (op) {
            case 0: return b_.OpResult(SpvOpLogicalAnd, t_bool_, {a, bv});
            case 1: return b_.OpResult(SpvOpLogicalOr, t_bool_, {a, bv});
            case 2: return b_.OpResult(SpvOpLogicalNotEqual, t_bool_, {a, bv});
            default: Error("bad boolean op"); return a;
        }
    }

    // ---- attributes ---------------------------------------------------------------------------------------
    Id BuiltinVar(Id* slot, SpvStorageClass sc, Id type, SpvBuiltIn bi, const char* name) {
        if (!*slot) {
            *slot = b_.Variable(b_.TypePointer(sc, type), sc);
            b_.Decorate(*slot, SpvDecorationBuiltIn, {uint32_t(bi)});
            b_.Name(*slot, name);
            interface_.push_back(*slot);
        }
        return *slot;
    }
    Id InputVar(uint32_t loc) {
        if (!in_[loc]) {
            Id type = t_v4f_;
            if (stage_ == Stage::Vertex && (opt_.int_attrib_mask >> loc & 1))
                type = (opt_.signed_attrib_mask >> loc & 1) ? t_v4i_ : t_v4u_;
            in_type_[loc] = type;
            in_[loc] = b_.Variable(b_.TypePointer(SpvStorageClassInput, type), SpvStorageClassInput);
            b_.Decorate(in_[loc], SpvDecorationLocation, {loc});
            char name[16];
            snprintf(name, sizeof(name), "in%u", loc);
            b_.Name(in_[loc], name);
            interface_.push_back(in_[loc]);
            info_.input_mask |= 1u << loc;
        }
        return in_[loc];
    }
    Id ReadInputComponent(uint32_t loc, uint32_t comp) {
        Id var = InputVar(loc);
        Id type = in_type_[loc];
        Id scalar = type == t_v4f_ ? t_f32_ : (type == t_v4i_ ? t_i32_ : t_u32_);
        Id p = b_.OpResult(SpvOpAccessChain, b_.TypePointer(SpvStorageClassInput, scalar), {var, b_.ConstUint(comp)});
        Id v = b_.OpResult(SpvOpLoad, scalar, {p});
        return scalar == t_u32_ ? v : BitcastU(v);
    }
    Id OutputVar(uint32_t loc) {
        if (!out_[loc]) {
            out_[loc] = b_.Variable(b_.TypePointer(SpvStorageClassOutput, t_v4f_), SpvStorageClassOutput);
            b_.Decorate(out_[loc], SpvDecorationLocation, {loc});
            char name[16];
            snprintf(name, sizeof(name), "out%u", loc);
            b_.Name(out_[loc], name);
            interface_.push_back(out_[loc]);
            info_.output_mask |= 1u << loc;
        }
        return out_[loc];
    }
    void StoreComponent(Id var, uint32_t comp, Id f) {
        Id p = b_.OpResult(SpvOpAccessChain, b_.TypePointer(SpvStorageClassOutput, t_f32_), {var, b_.ConstUint(comp)});
        b_.Op(SpvOpStore, {p, f});
    }

    // Vertex attribute load: a[0x80 + 16*i + 4*c] are generic inputs; 0x2f8/0x2fc instance/vertex id.
    Id LoadVertexAttribute(uint32_t addr) {
        if (addr >= 0x80 && addr < 0x280) return ReadInputComponent((addr - 0x80) / 16, (addr / 4) % 4);
        if (addr == 0x2fc) {
            Id v = BuiltinVar(&vertex_index_, SpvStorageClassInput, t_i32_, SpvBuiltInVertexIndex, "gl_VertexIndex");
            return BitcastU(b_.OpResult(SpvOpLoad, t_i32_, {v}));
        }
        if (addr == 0x2f8) {
            Id v = BuiltinVar(&instance_index_, SpvStorageClassInput, t_i32_, SpvBuiltInInstanceIndex,
                              "gl_InstanceIndex");
            return BitcastU(b_.OpResult(SpvOpLoad, t_i32_, {v}));
        }
        Error("unsupported vertex attribute load");
        return b_.ConstUint(0);
    }
    void StoreVertexAttribute(uint32_t addr, Id u) {
        Id f = BitcastF(u);
        if (addr >= 0x70 && addr < 0x80) {
            Id pos = BuiltinVar(&position_, SpvStorageClassOutput, t_v4f_, SpvBuiltInPosition, "gl_Position");
            StoreComponent(pos, (addr - 0x70) / 4, f);
            return;
        }
        if (addr >= 0x80 && addr < 0x280) {
            StoreComponent(OutputVar((addr - 0x80) / 16), (addr / 4) % 4, f);
            return;
        }
        if (addr == 0x6c) {
            Id ps = BuiltinVar(&point_size_, SpvStorageClassOutput, t_f32_, SpvBuiltInPointSize, "gl_PointSize");
            b_.Op(SpvOpStore, {ps, f});
            info_.writes_point_size = true;
            return;
        }
        Error("unsupported vertex attribute store");
    }

    // Fragment input. Maxwell interpolates attr/w and 1/w linearly and shaders divide explicitly
    // (IPA.PASS a[0x7c], MUFU.RCP, IPA.MULTIPLY). SPIR-V inputs are already perspective-correct, so we
    // report 1/w = 1.0 and the attribute itself, which keeps the shader's arithmetic exact.
    Id LoadFragmentAttribute(uint32_t addr, bool flat) {
        if (addr == 0x7c) return b_.ConstUint(0x3f800000);
        if (addr >= 0x70 && addr < 0x7c) {
            Id fc = BuiltinVar(&frag_coord_, SpvStorageClassInput, t_v4f_, SpvBuiltInFragCoord, "gl_FragCoord");
            Id p = b_.OpResult(SpvOpAccessChain, b_.TypePointer(SpvStorageClassInput, t_f32_),
                               {fc, b_.ConstUint((addr - 0x70) / 4)});
            return BitcastU(b_.OpResult(SpvOpLoad, t_f32_, {p}));
        }
        if (addr >= 0x80 && addr < 0x280) {
            uint32_t loc = (addr - 0x80) / 16;
            Id v = ReadInputComponent(loc, (addr / 4) % 4);
            if (flat && !(info_.flat_mask >> loc & 1)) {
                b_.Decorate(in_[loc], SpvDecorationFlat);
                info_.flat_mask |= 1u << loc;
            }
            return v;
        }
        if (addr == 0x2e0 || addr == 0x2e4) {
            Id pc = BuiltinVar(&point_coord_, SpvStorageClassInput, t_v2f_, SpvBuiltInPointCoord, "gl_PointCoord");
            Id p = b_.OpResult(SpvOpAccessChain, b_.TypePointer(SpvStorageClassInput, t_f32_),
                               {pc, b_.ConstUint((addr - 0x2e0) / 4)});
            return BitcastU(b_.OpResult(SpvOpLoad, t_f32_, {p}));
        }
        if (addr == 0x60) return b_.ConstUint(0);  // primitive id: not plumbed yet
        Error("unsupported fragment attribute");
        return b_.ConstUint(0);
    }

    // ---- exit ---------------------------------------------------------------------------------------------
    void EmitExit() {
        if (stage_ == Stage::Fragment) {
            // Outputs come from consecutive registers: for each render target, each enabled component.
            uint32_t omap, omap2;
            memcpy(&omap, code_ + kSphOffset + 18 * 4, 4);
            memcpy(&omap2, code_ + kSphOffset + 19 * 4, 4);
            // Alpha test (no fixed-function equivalent in Vulkan): compare RT0 alpha with the reference.
            if (opt_.alpha_func != 8 && (omap & 0x8)) {
                uint32_t alpha_reg = uint32_t(__builtin_popcount(omap & 0x7));
                Id alpha = GetF(alpha_reg);
                Id ref = AlphaRef();
                Id pass;
                switch (opt_.alpha_func) {
                    case 1: pass = b_.ConstBool(false); break;
                    case 2: pass = b_.OpResult(SpvOpFOrdLessThan, t_bool_, {alpha, ref}); break;
                    case 3: pass = b_.OpResult(SpvOpFOrdEqual, t_bool_, {alpha, ref}); break;
                    case 4: pass = b_.OpResult(SpvOpFOrdLessThanEqual, t_bool_, {alpha, ref}); break;
                    case 5: pass = b_.OpResult(SpvOpFOrdGreaterThan, t_bool_, {alpha, ref}); break;
                    case 6: pass = b_.OpResult(SpvOpFOrdNotEqual, t_bool_, {alpha, ref}); break;
                    case 7: pass = b_.OpResult(SpvOpFOrdGreaterThanEqual, t_bool_, {alpha, ref}); break;
                    default: pass = b_.ConstBool(true); break;
                }
                Id kill = b_.NewId(), cont = b_.NewId();
                b_.Op(SpvOpSelectionMerge, {cont, uint32_t(SpvSelectionControlMaskNone)});
                b_.Op(SpvOpBranchConditional, {pass, cont, kill});
                b_.Label(kill);
                b_.Op(SpvOpKill, {});
                b_.Label(cont);
                info_.uses_discard = true;
            }
            uint32_t reg = 0;
            for (uint32_t t = 0; t < 8; t++) {
                uint32_t mask = (omap >> (t * 4)) & 0xF;
                if (!mask) continue;
                Id var = OutputVar(t);
                for (uint32_t c = 0; c < 4; c++) {
                    if (mask >> c & 1)
                        StoreComponent(var, c, GetF(reg++));
                    else
                        StoreComponent(var, c, b_.ConstFloat(c == 3 ? 1.0f : 0.0f));
                }
            }
            if (omap2 & 2) {  // depth: one register past the sample-mask slot
                Id fd = BuiltinVar(&frag_depth_, SpvStorageClassOutput, t_f32_, SpvBuiltInFragDepth, "gl_FragDepth");
                b_.Op(SpvOpStore, {fd, GetF(reg + 1)});
                info_.writes_depth = true;
            }
        } else if (position_ && opt_.gl_depth_range) {
            // Clip z in [-w, w] -> [0, w].
            Id pz = b_.OpResult(SpvOpAccessChain, b_.TypePointer(SpvStorageClassOutput, t_f32_),
                                {position_, b_.ConstUint(2)});
            Id pw = b_.OpResult(SpvOpAccessChain, b_.TypePointer(SpvStorageClassOutput, t_f32_),
                                {position_, b_.ConstUint(3)});
            Id z = b_.OpResult(SpvOpLoad, t_f32_, {pz});
            Id w = b_.OpResult(SpvOpLoad, t_f32_, {pw});
            Id sum = b_.OpResult(SpvOpFAdd, t_f32_, {z, w});
            b_.Op(SpvOpStore, {pz, b_.OpResult(SpvOpFMul, t_f32_, {sum, b_.ConstFloat(0.5f)})});
        }
        b_.Op(SpvOpReturn, {});
    }

    Id AlphaRef() {
        if (!push_) {
            Id st = b_.TypeStruct({t_f32_});
            b_.Decorate(st, SpvDecorationBlock);
            b_.MemberDecorate(st, 0, SpvDecorationOffset, {0});
            push_ = b_.Variable(b_.TypePointer(SpvStorageClassPushConstant, st), SpvStorageClassPushConstant);
            b_.Name(push_, "push");
        }
        Id p = b_.OpResult(SpvOpAccessChain, b_.TypePointer(SpvStorageClassPushConstant, t_f32_), {push_, b_.ConstUint(0)});
        return b_.OpResult(SpvOpLoad, t_f32_, {p});
    }

    // ---- textures -------------------------------------------------------------------------------------------
    Id TextureVar(uint32_t slot, bool shadow, TexDim dim, bool array) {
        if (slot >= 32) {
            Error("texture slot out of range");
            slot = 0;
        }
        if (info_.texture_dim[slot] != TexDim::None &&
            (info_.texture_dim[slot] != dim || info_.texture_array[slot] != array)) {
            Error("texture slot used with two dimensions");
        }
        info_.texture_dim[slot] = dim;
        info_.texture_array[slot] = array;
        Id& var = shadow ? shadow_[slot] : tex_[slot];
        Id& sampled = shadow ? shadow_type_[slot] : tex_type_[slot];
        if (!var) {
            SpvDim sd = dim == TexDim::D1 ? SpvDim1D : dim == TexDim::D3 ? SpvDim3D : dim == TexDim::Cube ? SpvDimCube : SpvDim2D;
            if (sd == SpvDim1D) b_.Capability(SpvCapabilitySampled1D);
            if (sd == SpvDimCube && array) b_.Capability(SpvCapabilitySampledCubeArray);
            Id image = b_.TypeImage(t_f32_, sd, shadow ? 1 : 0, array ? 1 : 0);
            (shadow ? shadow_image_type_ : tex_image_type_)[slot] = image;
            sampled = b_.TypeSampledImage(image);
            var = b_.Variable(b_.TypePointer(SpvStorageClassUniformConstant, sampled), SpvStorageClassUniformConstant);
            b_.Decorate(var, SpvDecorationDescriptorSet, {DescriptorSet()});
            b_.Decorate(var, SpvDecorationBinding, {(shadow ? 64u : 32u) + slot});
            char name[24];
            snprintf(name, sizeof(name), shadow ? "shadow%u" : "tex%u", slot);
            b_.Name(var, name);
            if (shadow)
                info_.shadow_mask |= 1u << slot;
            else
                info_.texture_mask |= 1u << slot;
        }
        return var;
    }

    // TEX (bindless): dest 0-7, coords 8-15, meta 20-27, type 28-30 (bit 28 = array), mask 31-34,
    // aoffi 36, lod mode 37-39, depth compare 50. Meta registers: handle, [lod/bias], [offset], [dref].
    void EmitTexBindless(uint64_t i) {
        const uint32_t dest = uint32_t(Bits(i, 0, 8));
        uint32_t coord = uint32_t(Bits(i, 8, 8));
        uint32_t meta = uint32_t(Bits(i, 20, 8));
        const uint32_t type = uint32_t(Bits(i, 28, 3));
        const uint32_t mask = uint32_t(Bits(i, 31, 4));
        const bool aoffi = Bits(i, 36, 1);
        const uint32_t lod_mode = uint32_t(Bits(i, 37, 3));
        const bool dc = Bits(i, 50, 1);
        const bool array = type & 1;
        TexDim dim;
        uint32_t ncoords;
        switch (type >> 1) {
            case 0: dim = TexDim::D1; ncoords = 1; break;
            case 1: dim = TexDim::D2; ncoords = 2; break;
            case 2: dim = TexDim::D3; ncoords = 3; break;
            default: dim = TexDim::Cube; ncoords = 3; break;
        }
        const Known& hk = known_[meta];
        if (hk.kind != Known::TexSlot) {
            Error("texture handle not from the binding table");
            return;
        }
        const uint32_t slot = hk.value;
        meta++;
        Id var = TextureVar(slot, dc, dim, array);
        Id sampled_type = dc ? shadow_type_[slot] : tex_type_[slot];

        // Coordinates (array layer first, as an integer).
        std::vector<Id> comps;
        Id layer = 0;
        if (array) layer = b_.OpResult(SpvOpConvertUToF, t_f32_, {GetU(coord++)});
        for (uint32_t c = 0; c < ncoords; c++) comps.push_back(GetF(coord + c));
        if (layer) comps.push_back(layer);
        Id coords;
        if (comps.size() == 1) {
            coords = comps[0];
        } else {
            Id vt = comps.size() == 2 ? t_v2f_ : comps.size() == 3 ? t_v3f_ : t_v4f_;
            std::vector<uint32_t> w(comps.begin(), comps.end());
            coords = b_.OpResult(SpvOpCompositeConstruct, vt, w);
        }

        Id lod_value = 0;
        const bool has_lod_reg = lod_mode == 2 || lod_mode == 3 || lod_mode == 6 || lod_mode == 7;
        if (has_lod_reg) lod_value = GetF(meta++);
        uint32_t offset_operand_mask = 0;
        Id offset = 0;
        if (aoffi) {
            uint32_t packed;
            const uint32_t noff = dim == TexDim::D3 ? 3 : (dim == TexDim::D1 ? 1 : 2);
            Id ot = noff == 1 ? t_i32_ : noff == 2 ? t_v2i_ : t_v3i_;
            if (known_[meta].kind == Known::Const && dim != TexDim::Cube) {
                packed = known_[meta].value;
                std::vector<Id> parts;
                for (uint32_t c = 0; c < noff; c++) {
                    int32_t v = int32_t((packed >> (4 * c)) & 0xF);
                    if (v & 8) v -= 16;
                    parts.push_back(b_.ConstInt(v));
                }
                offset = noff == 1 ? parts[0]
                                   : (noff == 2 ? b_.ConstComposite(ot, {parts[0], parts[1]})
                                                : b_.ConstComposite(ot, {parts[0], parts[1], parts[2]}));
                offset_operand_mask = SpvImageOperandsConstOffsetMask;
            } else if (dim != TexDim::Cube) {
                b_.Capability(SpvCapabilityImageGatherExtended);
                Id raw = BitcastI(GetU(meta));
                std::vector<Id> parts;
                for (uint32_t c = 0; c < noff; c++)
                    parts.push_back(b_.OpResult(SpvOpBitFieldSExtract, t_i32_, {raw, b_.ConstUint(4 * c), b_.ConstUint(4)}));
                if (noff == 1) {
                    offset = parts[0];
                } else {
                    std::vector<uint32_t> w(parts.begin(), parts.end());
                    offset = b_.OpResult(SpvOpCompositeConstruct, ot, w);
                }
                offset_operand_mask = SpvImageOperandsOffsetMask;
            }
            meta++;
        }
        Id dref = dc ? GetF(meta++) : 0;

        Id image = b_.OpResult(SpvOpLoad, sampled_type, {var});
        const bool implicit = stage_ == Stage::Fragment && (lod_mode == 0 || lod_mode == 2 || lod_mode == 6);
        std::vector<uint32_t> args = {image, coords};
        if (dc) args.push_back(dref);
        uint32_t operands = offset_operand_mask;
        std::vector<uint32_t> extra;
        if (implicit) {
            if (lod_mode == 2 || lod_mode == 6) {
                operands |= SpvImageOperandsBiasMask;
                extra.push_back(lod_value);
            }
        } else {
            operands |= SpvImageOperandsLodMask;
            extra.push_back((lod_mode == 3 || lod_mode == 7) ? lod_value : b_.ConstFloat(0.0f));
        }
        if (offset_operand_mask) extra.push_back(offset);
        if (operands) {
            args.push_back(operands);
            args.insert(args.end(), extra.begin(), extra.end());
        }
        SpvOp op = dc ? (implicit ? SpvOpImageSampleDrefImplicitLod : SpvOpImageSampleDrefExplicitLod)
                      : (implicit ? SpvOpImageSampleImplicitLod : SpvOpImageSampleExplicitLod);
        Id result = b_.OpResult(op, dc ? t_f32_ : t_v4f_, args);
        uint32_t d = dest;
        for (uint32_t c = 0; c < 4; c++) {
            if (!(mask >> c & 1)) continue;
            Id v = dc ? result : b_.OpResult(SpvOpCompositeExtract, t_f32_, {result, c});
            if (d != 255) SetF(d++, v);
        }
    }

    static void TexTypeInfo(uint32_t type, TexDim* dim, uint32_t* ncoords) {
        switch (type >> 1) {
            case 0: *dim = TexDim::D1; *ncoords = 1; break;
            case 1: *dim = TexDim::D2; *ncoords = 2; break;
            case 2: *dim = TexDim::D3; *ncoords = 3; break;
            default: *dim = TexDim::Cube; *ncoords = 3; break;
        }
    }
    void WriteMasked(uint32_t dest, uint32_t mask, Id vec4_result) {
        uint32_t d = dest;
        for (uint32_t c = 0; c < 4; c++) {
            if (!(mask >> c & 1)) continue;
            Id v = b_.OpResult(SpvOpCompositeExtract, t_f32_, {vec4_result, c});
            if (d != 255) SetF(d++, v);
        }
    }

    // TLD (bindless texel fetch): integer coordinates, mip level 0.
    void EmitTexelFetchBindless(uint64_t i) {
        const uint32_t dest = uint32_t(Bits(i, 0, 8));
        uint32_t coord = uint32_t(Bits(i, 8, 8));
        const uint32_t meta = uint32_t(Bits(i, 20, 8));
        const uint32_t type = uint32_t(Bits(i, 28, 3)), mask = uint32_t(Bits(i, 31, 4));
        TexDim dim;
        uint32_t n;
        TexTypeInfo(type, &dim, &n);
        const bool array = type & 1;
        if (dim == TexDim::Cube) return Error("texel fetch from a cube map");
        if (known_[meta].kind != Known::TexSlot) return Error("texture handle not from the binding table");
        const uint32_t slot = known_[meta].value;
        Id var = TextureVar(slot, false, dim, array);
        std::vector<Id> c;
        Id layer = array ? BitcastI(GetU(coord++)) : 0;
        for (uint32_t k = 0; k < n; k++) c.push_back(BitcastI(GetU(coord + k)));
        if (layer) c.push_back(layer);
        Id coords = c[0];
        if (c.size() > 1) {
            Id vt = c.size() == 2 ? t_v2i_ : c.size() == 3 ? t_v3i_ : b_.TypeVector(t_i32_, 4);
            coords = b_.OpResult(SpvOpCompositeConstruct, vt, std::vector<uint32_t>(c.begin(), c.end()));
        }
        Id sampled = b_.OpResult(SpvOpLoad, tex_type_[slot], {var});
        Id image = b_.OpResult(SpvOpImage, tex_image_type_[slot], {sampled});
        Id r = b_.OpResult(SpvOpImageFetch, t_v4f_, {image, coords, uint32_t(SpvImageOperandsLodMask), b_.ConstInt(0)});
        WriteMasked(dest, mask, r);
    }

    // TLD4 (bindless gather): component 38-39, offset type 36-37 (1 = AOFFI), depth compare 50.
    void EmitGatherBindless(uint64_t i) {
        const uint32_t dest = uint32_t(Bits(i, 0, 8));
        uint32_t coord = uint32_t(Bits(i, 8, 8));
        uint32_t meta = uint32_t(Bits(i, 20, 8));
        const uint32_t type = uint32_t(Bits(i, 28, 3)), mask = uint32_t(Bits(i, 31, 4));
        const uint32_t offset_type = uint32_t(Bits(i, 36, 2)), component = uint32_t(Bits(i, 38, 2));
        const bool dc = Bits(i, 50, 1);
        TexDim dim;
        uint32_t n;
        TexTypeInfo(type, &dim, &n);
        const bool array = type & 1;
        if (dim != TexDim::D2 && dim != TexDim::Cube) return Error("gather from a 1D/3D texture");
        if (offset_type > 1) return Error("per-texel gather offsets");
        if (known_[meta].kind != Known::TexSlot) return Error("texture handle not from the binding table");
        const uint32_t slot = known_[meta].value;
        meta++;
        Id var = TextureVar(slot, dc, dim, array);
        std::vector<Id> c;
        Id layer = array ? b_.OpResult(SpvOpConvertUToF, t_f32_, {GetU(coord++)}) : 0;
        for (uint32_t k = 0; k < n; k++) c.push_back(GetF(coord + k));
        if (layer) c.push_back(layer);
        Id vt = c.size() == 2 ? t_v2f_ : c.size() == 3 ? t_v3f_ : t_v4f_;
        Id coords = b_.OpResult(SpvOpCompositeConstruct, vt, std::vector<uint32_t>(c.begin(), c.end()));
        std::vector<uint32_t> operands;
        uint32_t mask_bits = 0;
        if (offset_type == 1 && dim == TexDim::D2) {
            if (known_[meta].kind == Known::Const) {
                const uint32_t packed = known_[meta].value;
                int32_t ox = int32_t(packed & 0xF), oy = int32_t((packed >> 4) & 0xF);
                if (ox & 8) ox -= 16;
                if (oy & 8) oy -= 16;
                operands.push_back(b_.ConstComposite(t_v2i_, {b_.ConstInt(ox), b_.ConstInt(oy)}));
                mask_bits = SpvImageOperandsConstOffsetMask;
            } else {
                b_.Capability(SpvCapabilityImageGatherExtended);
                Id raw = BitcastI(GetU(meta));
                Id ox = b_.OpResult(SpvOpBitFieldSExtract, t_i32_, {raw, b_.ConstUint(0), b_.ConstUint(4)});
                Id oy = b_.OpResult(SpvOpBitFieldSExtract, t_i32_, {raw, b_.ConstUint(4), b_.ConstUint(4)});
                operands.push_back(b_.OpResult(SpvOpCompositeConstruct, t_v2i_, {ox, oy}));
                mask_bits = SpvImageOperandsOffsetMask;
            }
            meta++;
        }
        Id sampled = b_.OpResult(SpvOpLoad, dc ? shadow_type_[slot] : tex_type_[slot], {var});
        if (!dc && dim == TexDim::D2) {
            // Plain gathers are done as four texel fetches (clamp to edge), which is what a gather returns by
            // definition: Adreno returns zeros for gathers from 32-bit float textures.
            b_.Capability(SpvCapabilityImageQuery);
            Id image = b_.OpResult(SpvOpImage, tex_image_type_[slot], {sampled});
            Id size_any = b_.OpResult(SpvOpImageQuerySizeLod, array ? t_v3i_ : t_v2i_, {image, b_.ConstInt(0)});
            Id sx = b_.OpResult(SpvOpCompositeExtract, t_i32_, {size_any, 0});
            Id sy = b_.OpResult(SpvOpCompositeExtract, t_i32_, {size_any, 1});
            Id fx = b_.OpResult(SpvOpFSub, t_f32_, {b_.OpResult(SpvOpFMul, t_f32_, {c[0], b_.OpResult(SpvOpConvertSToF, t_f32_, {sx})}), b_.ConstFloat(0.5f)});
            Id fy = b_.OpResult(SpvOpFSub, t_f32_, {b_.OpResult(SpvOpFMul, t_f32_, {c[1], b_.OpResult(SpvOpConvertSToF, t_f32_, {sy})}), b_.ConstFloat(0.5f)});
            Id x0 = b_.OpResult(SpvOpConvertFToS, t_i32_, {b_.Glsl(t_f32_, G::kFloor, {fx})});
            Id y0 = b_.OpResult(SpvOpConvertFToS, t_i32_, {b_.Glsl(t_f32_, G::kFloor, {fy})});
            if (!operands.empty()) {  // texel offset
                Id off = operands[0];
                x0 = b_.OpResult(SpvOpIAdd, t_i32_, {x0, b_.OpResult(SpvOpCompositeExtract, t_i32_, {off, 0})});
                y0 = b_.OpResult(SpvOpIAdd, t_i32_, {y0, b_.OpResult(SpvOpCompositeExtract, t_i32_, {off, 1})});
            }
            Id mx = b_.OpResult(SpvOpISub, t_i32_, {sx, b_.ConstInt(1)});
            Id my = b_.OpResult(SpvOpISub, t_i32_, {sy, b_.ConstInt(1)});
            auto clampi = [&](Id v, Id hi) { return b_.Glsl(t_i32_, G::kSClamp, {v, b_.ConstInt(0), hi}); };
            Id xs[2] = {clampi(x0, mx), clampi(b_.OpResult(SpvOpIAdd, t_i32_, {x0, b_.ConstInt(1)}), mx)};
            Id ys[2] = {clampi(y0, my), clampi(b_.OpResult(SpvOpIAdd, t_i32_, {y0, b_.ConstInt(1)}), my)};
            Id li = array ? b_.OpResult(SpvOpConvertFToS, t_i32_, {layer}) : 0;
            // Gather order: (x0,y1), (x1,y1), (x1,y0), (x0,y0).
            const int order[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
            Id comps[4];
            for (int k = 0; k < 4; k++) {
                Id pc = array ? b_.OpResult(SpvOpCompositeConstruct, t_v3i_, {xs[order[k][0]], ys[order[k][1]], li})
                              : b_.OpResult(SpvOpCompositeConstruct, t_v2i_, {xs[order[k][0]], ys[order[k][1]]});
                Id t = b_.OpResult(SpvOpImageFetch, t_v4f_, {image, pc, uint32_t(SpvImageOperandsLodMask), b_.ConstInt(0)});
                comps[k] = b_.OpResult(SpvOpCompositeExtract, t_f32_, {t, component});
            }
            Id r = b_.OpResult(SpvOpCompositeConstruct, t_v4f_, {comps[0], comps[1], comps[2], comps[3]});
            WriteMasked(dest, mask, r);
            return;
        }
        std::vector<uint32_t> args = {sampled, coords};
        if (dc)
            args.push_back(GetF(meta++));
        else
            args.push_back(b_.ConstUint(component));
        if (mask_bits) {
            args.push_back(mask_bits);
            args.insert(args.end(), operands.begin(), operands.end());
        }
        Id r = b_.OpResult(dc ? SpvOpImageDrefGather : SpvOpImageGather, t_v4f_, args);
        WriteMasked(dest, mask, r);
    }

    // ---- forward branches (SYNC to an SSY point, BRA) as structured selections ------------------------------
    // Branch targets are relative to the next instruction. A target on a scheduling control word means the
    // instruction after it.
    uint32_t BranchTarget(uint64_t i) const {
        int32_t off = int32_t(Bits(i, 20, 24) << 8) >> 8;
        const int32_t signed_target = int32_t(pc_) + 8 + off;
        if (signed_target < 0) return 0;  // before the program: reported as a backward branch
        uint32_t t = uint32_t(signed_target);
        if (t % 32 == 0) t += 8;
        return t;
    }
    void JumpForward(uint32_t target, bool predicated, uint32_t guard) {
        // Blocks must nest: a new one may not outlive the enclosing one.
        if (!blocks_.empty() && target > blocks_.back().end) {
            Error("unstructured branch out of a conditional block");
            return;
        }
        if (predicated) {
            // Threads where the guard holds jump to the target: the code up to it runs only when it fails.
            Id skip = GetPred(guard & 7, guard & 8);
            Id run = b_.OpResult(SpvOpLogicalNot, t_bool_, {skip});
            Id body = b_.NewId(), merge = b_.NewId();
            b_.Op(SpvOpSelectionMerge, {merge, uint32_t(SpvSelectionControlMaskNone)});
            b_.Op(SpvOpBranchConditional, {run, body, merge});
            b_.Label(body);
            blocks_.push_back({merge, target, false, std::vector<Known>(known_, known_ + 256)});
            return;
        }
        // Everyone jumps: end the innermost block if it ends there, then skip the dead code.
        if (!blocks_.empty()) {
            if (blocks_.back().end != target) {
                Error("unstructured jump inside a conditional block");
                return;
            }
            if (!blocks_.back().closed) b_.Op(SpvOpBranch, {blocks_.back().merge});
            blocks_.back().closed = true;
        }
        skip_until_ = target;
    }

    // ---- surface stores (SUST.P to a bound image) ----------------------------------------------------------
    // data 0-7, coords 8-15, component mask 20-23 (P mode), type 33-35, bound 51 (image index 36-48 is a byte/4
    // offset into the driver's binding table: images start at 0x120, 8 bytes each), D mode 52.
    void EmitSurfaceStore(uint64_t i) {
        const uint32_t data = uint32_t(Bits(i, 0, 8)), coord = uint32_t(Bits(i, 8, 8));
        const uint32_t mask = uint32_t(Bits(i, 20, 4)), type = uint32_t(Bits(i, 33, 3));
        if (Bits(i, 52, 1)) return Error("raw (SUST.D) surface stores");
        if (!Bits(i, 51, 1)) return Error("bindless surface stores");
        const uint32_t table = uint32_t(Bits(i, 36, 13)) * 4;
        if (table < 0x120 || (table - 0x120) % 8 || (table - 0x120) / 8 >= 8) return Error("unexpected image binding offset");
        const uint32_t slot = (table - 0x120) / 8;
        TexDim dim;
        bool array = false;
        uint32_t n;
        switch (type) {
            case 0: dim = TexDim::D1; n = 1; break;
            case 2: dim = TexDim::D1; n = 1; array = true; break;
            case 3: dim = TexDim::D2; n = 2; break;
            case 4: dim = TexDim::D2; n = 2; array = true; break;
            case 5: dim = TexDim::D3; n = 3; break;
            default: return Error("unsupported surface type");
        }
        if (info_.image_dim[slot] != TexDim::None && (info_.image_dim[slot] != dim || info_.image_array[slot] != array))
            return Error("image slot used with two dimensions");
        info_.image_dim[slot] = dim;
        info_.image_array[slot] = array;
        if (!image_[slot]) {
            b_.Capability(SpvCapabilityStorageImageWriteWithoutFormat);
            if (dim == TexDim::D1) b_.Capability(SpvCapabilityImage1D);
            SpvDim sd = dim == TexDim::D1 ? SpvDim1D : dim == TexDim::D3 ? SpvDim3D : SpvDim2D;
            image_type_[slot] = b_.TypeStorageImage(t_f32_, sd, array ? 1 : 0);
            image_[slot] = b_.Variable(b_.TypePointer(SpvStorageClassUniformConstant, image_type_[slot]),
                                       SpvStorageClassUniformConstant);
            b_.Decorate(image_[slot], SpvDecorationDescriptorSet, {DescriptorSet()});
            b_.Decorate(image_[slot], SpvDecorationBinding, {96 + slot});
            b_.Decorate(image_[slot], SpvDecorationNonReadable);
            char name[16];
            snprintf(name, sizeof(name), "img%u", slot);
            b_.Name(image_[slot], name);
            info_.image_mask |= 1u << slot;
        }
        std::vector<uint32_t> c;
        for (uint32_t k = 0; k < n + (array ? 1 : 0); k++) c.push_back(BitcastI(GetU(coord + k)));
        Id coords = c.size() == 1 ? c[0]
                                  : b_.OpResult(SpvOpCompositeConstruct, c.size() == 2 ? t_v2i_ : t_v3i_, c);
        std::vector<uint32_t> texel;
        uint32_t r = data;
        for (uint32_t k = 0; k < 4; k++)
            texel.push_back((mask >> k & 1) ? GetF(r++) : b_.ConstFloat(k == 3 ? 1.0f : 0.0f));
        Id value = b_.OpResult(SpvOpCompositeConstruct, t_v4f_, texel);
        Id img = b_.OpResult(SpvOpLoad, image_type_[slot], {image_[slot]});
        b_.Op(SpvOpImageWrite, {img, coords, value});
    }

    // ---- local memory (STL / LDL): a per-invocation 4 KiB array ------------------------------------------
    Id LocalWord(Id byte_address) {
        if (!local_) {
            Id arr = b_.TypeArray(t_u32_, 1024);
            local_ = b_.LocalVariable(b_.TypePointer(SpvStorageClassFunction, arr));
            b_.Name(local_, "lmem");
        }
        Id idx = b_.OpResult(SpvOpShiftRightLogical, t_u32_, {byte_address, b_.ConstUint(2)});
        idx = b_.OpResult(SpvOpBitwiseAnd, t_u32_, {idx, b_.ConstUint(1023)});
        return b_.OpResult(SpvOpAccessChain, p_fn_u32_, {local_, idx});
    }
    void EmitLocal(uint64_t i, bool store) {
        const uint32_t reg = uint32_t(Bits(i, 0, 8)), addr = uint32_t(Bits(i, 8, 8));
        const int32_t off = int32_t(Bits(i, 20, 24) << 8) >> 8;
        const uint32_t size = uint32_t(Bits(i, 48, 3));
        const uint32_t words = size == 4 ? 1 : size == 5 ? 2 : size == 6 ? 4 : 0;
        if (!words) return Error("unsupported local memory access size");
        Id base = b_.OpResult(SpvOpIAdd, t_u32_, {GetU(addr), b_.ConstUint(uint32_t(off))});
        for (uint32_t k = 0; k < words; k++) {
            Id a = k ? b_.OpResult(SpvOpIAdd, t_u32_, {base, b_.ConstUint(4 * k)}) : base;
            Id p = LocalWord(a);
            if (store)
                b_.Op(SpvOpStore, {p, GetU(reg + k)});
            else
                SetU(reg + k, b_.OpResult(SpvOpLoad, t_u32_, {p}));
        }
    }

    // ---- XMAD (16x16 multiply-add) ------------------------------------------------------------------------
    void EmitXmad(uint64_t i, Form f) {
        const uint32_t dest = uint32_t(Bits(i, 0, 8));
        Id a = GetU(uint32_t(Bits(i, 8, 8)));
        bool half_a = Bits(i, 53, 1), signed_a = Bits(i, 48, 1), signed_b = Bits(i, 49, 1);
        bool half_b = false, psl = false, mrg = false;
        uint32_t mode = 0;
        Id bv, c;
        switch (f) {
            case Form::Imm:
                bv = b_.ConstUint(uint32_t(Bits(i, 20, 16)));
                psl = Bits(i, 36, 1);
                mrg = Bits(i, 37, 1);
                mode = uint32_t(Bits(i, 50, 3));
                c = GetU(uint32_t(Bits(i, 39, 8)));
                break;
            case Form::Reg:
                bv = GetU(uint32_t(Bits(i, 20, 8)));
                half_b = Bits(i, 35, 1);
                psl = Bits(i, 36, 1);
                mrg = Bits(i, 37, 1);
                mode = uint32_t(Bits(i, 50, 3));
                c = GetU(uint32_t(Bits(i, 39, 8)));
                break;
            case Form::RC:
                bv = GetU(uint32_t(Bits(i, 39, 8)));
                c = Cbuf(uint32_t(Bits(i, 34, 5)), uint32_t(Bits(i, 20, 14) * 4));
                half_b = Bits(i, 52, 1);
                mode = uint32_t(Bits(i, 50, 2));
                break;
            default:  // CR
                bv = Cbuf(uint32_t(Bits(i, 34, 5)), uint32_t(Bits(i, 20, 14) * 4));
                c = GetU(uint32_t(Bits(i, 39, 8)));
                half_b = Bits(i, 52, 1);
                psl = Bits(i, 55, 1);
                mrg = Bits(i, 56, 1);
                mode = uint32_t(Bits(i, 50, 2));
                break;
        }
        auto half = [&](Id v, bool hi, bool sgn) {
            if (sgn)
                return BitcastU(b_.OpResult(SpvOpBitFieldSExtract, t_i32_, {BitcastI(v), b_.ConstUint(hi ? 16 : 0), b_.ConstUint(16)}));
            return b_.OpResult(SpvOpBitFieldUExtract, t_u32_, {v, b_.ConstUint(hi ? 16 : 0), b_.ConstUint(16)});
        };
        Id op_a = half(a, half_a, signed_a);
        Id op_b = f == Form::Imm ? bv : half(bv, half_b, signed_b);
        Id product = b_.OpResult(SpvOpIMul, t_u32_, {op_a, op_b});
        if (psl) product = b_.OpResult(SpvOpShiftLeftLogical, t_u32_, {product, b_.ConstUint(16)});
        Id op_c;
        switch (mode) {
            case 0: op_c = c; break;                                                                          // default
            case 1: op_c = b_.OpResult(SpvOpBitwiseAnd, t_u32_, {c, b_.ConstUint(0xFFFF)}); break;            // CLO
            case 2: op_c = b_.OpResult(SpvOpShiftRightLogical, t_u32_, {c, b_.ConstUint(16)}); break;         // CHI
            case 4: {                                                                                          // CBCC
                Id bs = b_.OpResult(SpvOpShiftLeftLogical, t_u32_, {bv, b_.ConstUint(16)});
                op_c = b_.OpResult(SpvOpIAdd, t_u32_, {c, bs});
                break;
            }
            default: Error("unsupported XMAD mode"); op_c = c; break;
        }
        Id result = b_.OpResult(SpvOpIAdd, t_u32_, {product, op_c});
        if (mrg) {
            Id lo = b_.OpResult(SpvOpBitwiseAnd, t_u32_, {result, b_.ConstUint(0xFFFF)});
            Id hi = b_.OpResult(SpvOpShiftLeftLogical, t_u32_, {bv, b_.ConstUint(16)});
            result = b_.OpResult(SpvOpBitwiseOr, t_u32_, {lo, hi});
        }
        SetU(dest, result);
    }

    // ---- instruction emission -------------------------------------------------------------------------------
    // Returns true when translation is finished (unconditional EXIT or KIL).
    bool Emit(uint64_t i) {
        Decoded d = Patterns().Decode(i);
        const uint32_t guard = uint32_t(Bits(i, 16, 4));
        const bool predicated = (guard & 7) != 7 || (guard & 8);
        if (guard == 0xF) return false;  // @!PT: never executes

        // Instructions that change control flow or end the program.
        switch (d.op) {
            case Op::EXIT:
            case Op::KIL:
                if (!predicated) {
                    if (d.op == Op::EXIT) {
                        EmitExit();
                    } else {
                        info_.uses_discard = true;
                        b_.Op(SpvOpKill, {});
                    }
                    if (blocks_.empty()) return true;
                    // Inside a conditional block: the rest of the block is dead.
                    blocks_.back().closed = true;
                    skip_until_ = blocks_.back().end;
                    return false;
                }
                break;
            case Op::SSY:
                // Pushes the reconvergence point for a following SYNC.
                ssy_.push_back(BranchTarget(i));
                return false;
            case Op::BRA: {
                const uint32_t target = BranchTarget(i);
                if (target <= pc_) {
                    // A branch to itself after the final EXIT is padding; anything else is a loop.
                    if (target == pc_ && !predicated) return blocks_.empty();
                    Error("backward branches (loops) are not supported yet");
                    return false;
                }
                JumpForward(target, predicated, guard);
                return false;
            }
            case Op::SYNC: {
                if (ssy_.empty()) {
                    Error("SYNC without SSY");
                    return false;
                }
                const uint32_t target = ssy_.back();
                if (target <= pc_) {
                    Error("backward SYNC target");
                    return false;
                }
                JumpForward(target, predicated, guard);
                return false;
            }
            case Op::PBK: case Op::BRK: case Op::PCNT: case Op::CONT:
                Error(std::string("control flow (") + OpName(d.op) + ") is not supported yet");
                return false;
            case Op::DEPBAR: case Op::NOP: case Op::MEMBAR:
                return false;
            default: break;
        }

        Id merge = 0;
        if (predicated) {
            Id cond = GetPred(guard & 7, guard & 8);
            Id body = b_.NewId();
            merge = b_.NewId();
            b_.Op(SpvOpSelectionMerge, {merge, uint32_t(SpvSelectionControlMaskNone)});
            b_.Op(SpvOpBranchConditional, {cond, body, merge});
            b_.Label(body);
            predicated_ = true;
        }
        bool terminated = false;
        switch (d.op) {
            case Op::EXIT: EmitExit(); terminated = true; break;
            case Op::KIL: info_.uses_discard = true; b_.Op(SpvOpKill, {}); terminated = true; break;
            default: EmitOp(i, d); break;
        }
        if (predicated) {
            if (!terminated) b_.Op(SpvOpBranch, {merge});
            b_.Label(merge);
            predicated_ = false;
        }
        return false;
    }

    void EmitOp(uint64_t i, Decoded d) {
        const uint32_t dest = uint32_t(Bits(i, 0, 8));
        const uint32_t ra = uint32_t(Bits(i, 8, 8));
        const Form f = d.form;
        switch (d.op) {
            case Op::MOV: {
                if (f == Form::Cbuf) {
                    const uint32_t idx = uint32_t(Bits(i, 34, 5)), off = uint32_t(Bits(i, 20, 14) * 4);
                    if (idx == kTextureTableCbuf && off >= kTextureTableBase && off < kTextureTableBase + 32 * 8 &&
                        (off - kTextureTableBase) % 8 == 0) {
                        // Bindless handle from the driver's texture-binding table: only its slot matters,
                        // so no constant buffer is needed for it.
                        const uint32_t slot = (off - kTextureTableBase) / 8;
                        SetU(dest, b_.ConstUint(slot), {Known::TexSlot, slot});
                    } else {
                        SetU(dest, Cbuf(idx, off));
                    }
                } else {
                    Known k;
                    if (f == Form::Imm) k = {Known::Const, Imm20Int(i)};
                    if (f == Form::Reg) k = known_[Bits(i, 20, 8)];
                    SetU(dest, SrcBU(i, f, false), k);
                }
                break;
            }
            case Op::MOV32I: {
                uint32_t v = uint32_t(Bits(i, 20, 32));
                SetU(dest, b_.ConstUint(v), {Known::Const, v});
                break;
            }
            case Op::FMUL: {
                Id a = GetF(ra);
                Id bv = FMod(SrcBF(i, f), false, Bits(i, 48, 1));
                Id r = b_.OpResult(SpvOpFMul, t_f32_, {a, bv});
                static const float kScale[8] = {1.0f, 0.5f, 0.25f, 0.125f, 8.0f, 4.0f, 2.0f, 1.0f};
                uint32_t scale = uint32_t(Bits(i, 41, 3));
                if (scale) r = b_.OpResult(SpvOpFMul, t_f32_, {r, b_.ConstFloat(kScale[scale])});
                SetF(dest, Sat(r, Bits(i, 50, 1)));
                break;
            }
            case Op::FMUL32I: {
                Id r = b_.OpResult(SpvOpFMul, t_f32_, {GetF(ra), b_.ConstFloatBits(uint32_t(Bits(i, 20, 32)))});
                SetF(dest, Sat(r, Bits(i, 55, 1)));
                break;
            }
            case Op::FADD: {
                Id a = FMod(GetF(ra), Bits(i, 46, 1), Bits(i, 48, 1));
                Id bv = FMod(SrcBF(i, f), Bits(i, 49, 1), Bits(i, 45, 1));
                SetF(dest, Sat(b_.OpResult(SpvOpFAdd, t_f32_, {a, bv}), Bits(i, 50, 1)));
                break;
            }
            case Op::FADD32I: {
                Id a = FMod(GetF(ra), Bits(i, 54, 1), Bits(i, 56, 1));
                Id bv = FMod(b_.ConstFloatBits(uint32_t(Bits(i, 20, 32))), Bits(i, 57, 1), Bits(i, 53, 1));
                SetF(dest, b_.OpResult(SpvOpFAdd, t_f32_, {a, bv}));
                break;
            }
            case Op::FFMA: {
                Id a = GetF(ra);
                Id bv = FMod(SrcBF(i, f), false, Bits(i, 48, 1));
                Id c = FMod(SrcCF(i, f), false, Bits(i, 49, 1));
                SetF(dest, Sat(b_.Glsl(t_f32_, G::kFma, {a, bv, c}), Bits(i, 50, 1)));
                break;
            }
            case Op::FFMA32I: {
                Id a = FMod(GetF(ra), false, Bits(i, 56, 1));
                Id c = FMod(GetF(dest), false, Bits(i, 57, 1));
                Id r = b_.Glsl(t_f32_, G::kFma, {a, b_.ConstFloatBits(uint32_t(Bits(i, 20, 32))), c});
                SetF(dest, Sat(r, Bits(i, 55, 1)));
                break;
            }
            case Op::FMNMX: {
                Id a = FMod(GetF(ra), Bits(i, 46, 1), Bits(i, 48, 1));
                Id bv = FMod(SrcBF(i, f), Bits(i, 49, 1), Bits(i, 45, 1));
                Id p = GetPred(uint32_t(Bits(i, 39, 3)), Bits(i, 42, 1));
                Id mn = b_.Glsl(t_f32_, G::kFMin, {a, bv});
                Id mx = b_.Glsl(t_f32_, G::kFMax, {a, bv});
                SetF(dest, b_.OpResult(SpvOpSelect, t_f32_, {p, mn, mx}));
                break;
            }
            case Op::MUFU: {
                Id a = FMod(GetF(ra), Bits(i, 46, 1), Bits(i, 48, 1));
                Id r;
                switch (Bits(i, 20, 4)) {
                    case 0: r = b_.Glsl(t_f32_, G::kCos, {a}); break;
                    case 1: r = b_.Glsl(t_f32_, G::kSin, {a}); break;
                    case 2: r = b_.Glsl(t_f32_, G::kExp2, {a}); break;
                    case 3: r = b_.Glsl(t_f32_, G::kLog2, {a}); break;
                    case 4: r = b_.OpResult(SpvOpFDiv, t_f32_, {b_.ConstFloat(1.0f), a}); break;
                    case 5: r = b_.Glsl(t_f32_, G::kInverseSqrt, {a}); break;
                    case 8: r = b_.Glsl(t_f32_, G::kSqrt, {a}); break;
                    default: Error("unsupported MUFU operation"); return;
                }
                SetF(dest, Sat(r, Bits(i, 50, 1)));
                break;
            }
            case Op::RRO:  // range reduction before MUFU.SIN/COS/EX2: identity at our precision
                SetF(dest, FMod(SrcBF(i, f), Bits(i, 49, 1), Bits(i, 45, 1)));
                break;
            case Op::FSET: {
                Id a = FMod(GetF(ra), Bits(i, 54, 1), Bits(i, 43, 1));
                Id bv = FMod(SrcBF(i, f), Bits(i, 44, 1), Bits(i, 53, 1));
                Id cmp = FCompare(uint32_t(Bits(i, 48, 4)), a, bv);
                Id r = BoolOp(uint32_t(Bits(i, 45, 2)), cmp, GetPred(uint32_t(Bits(i, 39, 3)), Bits(i, 42, 1)));
                Id t = Bits(i, 52, 1) ? b_.ConstUint(0x3f800000) : b_.ConstUint(0xFFFFFFFF);
                SetU(dest, b_.OpResult(SpvOpSelect, t_u32_, {r, t, b_.ConstUint(0)}));
                break;
            }
            case Op::FSETP: {
                Id a = FMod(GetF(ra), Bits(i, 7, 1), Bits(i, 43, 1));
                Id bv = FMod(SrcBF(i, f), Bits(i, 44, 1), Bits(i, 6, 1));
                Id cmp = FCompare(uint32_t(Bits(i, 48, 4)), a, bv);
                Id p = GetPred(uint32_t(Bits(i, 39, 3)), Bits(i, 42, 1));
                uint32_t bop = uint32_t(Bits(i, 45, 2));
                SetPred(uint32_t(Bits(i, 3, 3)), BoolOp(bop, cmp, p));
                SetPred(uint32_t(Bits(i, 0, 3)), BoolOp(bop, b_.OpResult(SpvOpLogicalNot, t_bool_, {cmp}), p));
                break;
            }
            case Op::ISETP: {
                Id a = GetU(ra), bv = SrcBU(i, f, false);
                Id cmp = ICompare(uint32_t(Bits(i, 49, 3)), Bits(i, 48, 1), a, bv);
                Id p = GetPred(uint32_t(Bits(i, 39, 3)), Bits(i, 42, 1));
                uint32_t bop = uint32_t(Bits(i, 45, 2));
                SetPred(uint32_t(Bits(i, 3, 3)), BoolOp(bop, cmp, p));
                SetPred(uint32_t(Bits(i, 0, 3)), BoolOp(bop, b_.OpResult(SpvOpLogicalNot, t_bool_, {cmp}), p));
                break;
            }
            case Op::ISET: {
                Id a = GetU(ra), bv = SrcBU(i, f, false);
                Id cmp = ICompare(uint32_t(Bits(i, 49, 3)), Bits(i, 48, 1), a, bv);
                Id r = BoolOp(uint32_t(Bits(i, 45, 2)), cmp, GetPred(uint32_t(Bits(i, 39, 3)), Bits(i, 42, 1)));
                Id t = Bits(i, 44, 1) ? b_.ConstUint(0x3f800000) : b_.ConstUint(0xFFFFFFFF);
                SetU(dest, b_.OpResult(SpvOpSelect, t_u32_, {r, t, b_.ConstUint(0)}));
                break;
            }
            case Op::PSETP: {
                // dest_b 0-2, dest_a 3-5, pred_a 12-14 (neg 15), pred_b 29-31 (neg 32), bop_1 24-25,
                // pred_c 39-41 (neg 42), bop_2 45-46.
                Id pa = GetPred(uint32_t(Bits(i, 12, 3)), Bits(i, 15, 1));
                Id pb = GetPred(uint32_t(Bits(i, 29, 3)), Bits(i, 32, 1));
                Id pc = GetPred(uint32_t(Bits(i, 39, 3)), Bits(i, 42, 1));
                Id lhs = BoolOp(uint32_t(Bits(i, 24, 2)), pa, pb);
                uint32_t bop2 = uint32_t(Bits(i, 45, 2));
                SetPred(uint32_t(Bits(i, 3, 3)), BoolOp(bop2, lhs, pc));
                SetPred(uint32_t(Bits(i, 0, 3)), BoolOp(bop2, b_.OpResult(SpvOpLogicalNot, t_bool_, {lhs}), pc));
                break;
            }
            case Op::SEL: {
                Id p = GetPred(uint32_t(Bits(i, 39, 3)), Bits(i, 42, 1));
                SetU(dest, b_.OpResult(SpvOpSelect, t_u32_, {p, GetU(ra), SrcBU(i, f, false)}));
                break;
            }
            case Op::IPA: {
                const uint32_t addr = uint32_t(Bits(i, 28, 10));
                const uint32_t mode = uint32_t(Bits(i, 54, 2));
                if (Bits(i, 38, 1)) {
                    Error("indexed IPA is not supported");
                    return;
                }
                Id v = LoadFragmentAttribute(addr, mode == 2);
                Id fv = BitcastF(v);
                if (mode == 1) fv = b_.OpResult(SpvOpFMul, t_f32_, {fv, GetF(uint32_t(Bits(i, 20, 8)))});
                SetF(dest, Sat(fv, Bits(i, 51, 1)));
                break;
            }
            case Op::ALD: {
                if (stage_ != Stage::Vertex || ra != 255 || Bits(i, 32, 1)) {
                    Error("unsupported ALD form");
                    return;
                }
                const uint32_t n = uint32_t(Bits(i, 47, 2)) + 1, addr = uint32_t(Bits(i, 20, 10));
                for (uint32_t k = 0; k < n; k++) SetU(dest + k, LoadVertexAttribute(addr + 4 * k));
                break;
            }
            case Op::AST: {
                if (stage_ != Stage::Vertex || ra != 255) {
                    Error("unsupported AST form");
                    return;
                }
                const uint32_t n = uint32_t(Bits(i, 47, 2)) + 1, addr = uint32_t(Bits(i, 20, 10));
                for (uint32_t k = 0; k < n; k++) StoreVertexAttribute(addr + 4 * k, GetU(dest + k));
                break;
            }
            case Op::LDC: {
                const uint32_t idx = uint32_t(Bits(i, 36, 5));
                const int32_t off = int32_t(int16_t(Bits(i, 20, 16)));
                const uint32_t size = uint32_t(Bits(i, 48, 3));
                if (Bits(i, 44, 2) != 0 || (size != 4 && size != 5 && size != 6)) {
                    Error("unsupported LDC form");
                    return;
                }
                Id base = b_.OpResult(SpvOpIAdd, t_u32_, {GetU(ra), b_.ConstUint(uint32_t(off))});
                const uint32_t words = size == 4 ? 1 : size == 5 ? 2 : 4;
                for (uint32_t k = 0; k < words; k++) {
                    Id addr = k ? b_.OpResult(SpvOpIAdd, t_u32_, {base, b_.ConstUint(4 * k)}) : base;
                    SetU(dest + k, CbufDynamic(idx, addr));
                }
                break;
            }
            case Op::TEXB: EmitTexBindless(i); break;
            case Op::TLDB: EmitTexelFetchBindless(i); break;
            case Op::TLD4B: EmitGatherBindless(i); break;
            case Op::STL: EmitLocal(i, true); break;
            case Op::SUST: EmitSurfaceStore(i); break;
            case Op::LDL: EmitLocal(i, false); break;
            case Op::XMAD: EmitXmad(i, f); break;
            case Op::I2F: {
                const uint32_t dst_fmt = uint32_t(Bits(i, 8, 2)), src_fmt = uint32_t(Bits(i, 10, 2));
                const bool sgn = Bits(i, 13, 1);
                const uint32_t sel = uint32_t(Bits(i, 41, 2));
                if (dst_fmt != 2 || src_fmt == 3) {
                    Error("unsupported I2F format");
                    return;
                }
                Id v = SrcBU(i, f, false);
                if (src_fmt != 2) {
                    uint32_t width = src_fmt == 0 ? 8 : 16, shift = sel * (src_fmt == 0 ? 8 : 16);
                    v = sgn ? BitcastU(b_.OpResult(SpvOpBitFieldSExtract, t_i32_, {BitcastI(v), b_.ConstUint(shift), b_.ConstUint(width)}))
                            : b_.OpResult(SpvOpBitFieldUExtract, t_u32_, {v, b_.ConstUint(shift), b_.ConstUint(width)});
                }
                Id r = sgn ? b_.OpResult(SpvOpConvertSToF, t_f32_, {BitcastI(v)}) : b_.OpResult(SpvOpConvertUToF, t_f32_, {v});
                SetF(dest, FMod(r, Bits(i, 49, 1), Bits(i, 45, 1)));
                break;
            }
            case Op::F2I: {
                // dst size 8-9 (0: 8, 1: 16, 2: 32 bit), src size 10-11 (2: F32), signed 12; saturates.
                const uint32_t dst = uint32_t(Bits(i, 8, 2));
                const bool sgn = Bits(i, 12, 1);
                if (dst == 3 || Bits(i, 10, 2) != 2) {
                    Error("unsupported F2I format");
                    return;
                }
                Id v = FMod(SrcBF(i, f), Bits(i, 49, 1), Bits(i, 45, 1));
                static const G kRound[4] = {G::kRoundEven, G::kFloor, G::kCeil, G::kTrunc};
                v = b_.Glsl(t_f32_, kRound[Bits(i, 39, 2)], {v});
                const uint32_t bits = 8u << dst;
                const double lo = sgn ? -double(1ull << (bits - 1)) : 0.0;
                const double hi = sgn ? double((1ull << (bits - 1)) - 1) : double((1ull << bits) - 1);
                // Clamp in float (the largest 32-bit values are not representable; stay just inside).
                const float flo = float(lo), fhi = bits == 32 ? (sgn ? 2147483520.0f : 4294967040.0f) : float(hi);
                v = b_.Glsl(t_f32_, G::kFClamp, {v, b_.ConstFloat(flo), b_.ConstFloat(fhi)});
                Id r = sgn ? BitcastU(b_.OpResult(SpvOpConvertFToS, t_i32_, {v}))
                           : b_.OpResult(SpvOpConvertFToU, t_u32_, {v});
                SetU(dest, r);
                break;
            }
            case Op::I2I: {
                // dst size 8-9, src size 10-11, dst signed 12, src signed 13, byte/half selector 41-42,
                // neg 45, abs 49, saturate 50.
                const uint32_t dsz = uint32_t(Bits(i, 8, 2)), ssz = uint32_t(Bits(i, 10, 2));
                const bool dsgn = Bits(i, 12, 1), ssgn = Bits(i, 13, 1), sat = Bits(i, 50, 1);
                if (dsz == 3 || ssz == 3) {
                    Error("unsupported I2I format");
                    return;
                }
                Id v = SrcBU(i, f, false);
                if (ssz != 2) {
                    const uint32_t width = 8u << ssz, shift = uint32_t(Bits(i, 41, 2)) * width;
                    v = ssgn ? BitcastU(b_.OpResult(SpvOpBitFieldSExtract, t_i32_, {BitcastI(v), b_.ConstUint(shift), b_.ConstUint(width)}))
                             : b_.OpResult(SpvOpBitFieldUExtract, t_u32_, {v, b_.ConstUint(shift), b_.ConstUint(width)});
                }
                if (Bits(i, 49, 1)) v = BitcastU(b_.Glsl(t_i32_, G::kSAbsShim, {BitcastI(v)}));
                if (Bits(i, 45, 1)) v = BitcastU(b_.OpResult(SpvOpSNegate, t_i32_, {BitcastI(v)}));
                const uint32_t bits = 8u << dsz;
                if (sat && bits < 32) {
                    if (ssgn) {
                        const int32_t lo = dsgn ? -(1 << (bits - 1)) : 0;
                        const int32_t hi = dsgn ? (1 << (bits - 1)) - 1 : int32_t((1u << bits) - 1);
                        v = BitcastU(b_.Glsl(t_i32_, G::kSClamp, {BitcastI(v), b_.ConstInt(lo), b_.ConstInt(hi)}));
                    } else {
                        const uint32_t hi = dsgn ? (1u << (bits - 1)) - 1 : (1u << bits) - 1;
                        v = b_.Glsl(t_u32_, G::kUMin, {v, b_.ConstUint(hi)});
                    }
                }
                if (bits < 32)
                    v = dsgn ? BitcastU(b_.OpResult(SpvOpBitFieldSExtract, t_i32_, {BitcastI(v), b_.ConstUint(0), b_.ConstUint(bits)}))
                             : b_.OpResult(SpvOpBitFieldUExtract, t_u32_, {v, b_.ConstUint(0), b_.ConstUint(bits)});
                SetU(dest, v);
                break;
            }
            case Op::F2F: {
                if (Bits(i, 8, 2) != 2 || Bits(i, 10, 2) != 2) {
                    Error("unsupported F2F format");
                    return;
                }
                Id v = FMod(SrcBF(i, f), Bits(i, 49, 1), Bits(i, 45, 1));
                if (Bits(i, 42, 1)) {  // round to integer
                    static const G kRound[4] = {G::kRoundEven, G::kFloor, G::kCeil, G::kTrunc};
                    v = b_.Glsl(t_f32_, kRound[Bits(i, 39, 2)], {v});
                }
                SetF(dest, Sat(v, Bits(i, 50, 1)));
                break;
            }
            case Op::IADD: {
                Id a = GetU(ra), bv = SrcBU(i, f, false);
                if (Bits(i, 49, 1)) a = BitcastU(b_.OpResult(SpvOpSNegate, t_i32_, {BitcastI(a)}));
                if (Bits(i, 48, 1)) bv = BitcastU(b_.OpResult(SpvOpSNegate, t_i32_, {BitcastI(bv)}));
                SetU(dest, b_.OpResult(SpvOpIAdd, t_u32_, {a, bv}));
                break;
            }
            case Op::IADD32I: {
                Id a = GetU(ra);
                if (Bits(i, 56, 1)) a = BitcastU(b_.OpResult(SpvOpSNegate, t_i32_, {BitcastI(a)}));
                SetU(dest, b_.OpResult(SpvOpIAdd, t_u32_, {a, b_.ConstUint(uint32_t(Bits(i, 20, 32)))}));
                break;
            }
            case Op::IADD3: {
                // dest = ((a + b) shifted) + c. Register form: halves a 35-36, b 33-34, c 31-32 (0 all, 1 low, 2
                // high), shift 37-38 (1: >> 16, 2: << 16) applied to a + b. All forms: negate a 51, b 50, c 49.
                Id a = GetU(ra), bv = SrcBU(i, f, false), c = GetU(uint32_t(Bits(i, 39, 8)));
                uint32_t shift = 0;
                if (f == Form::Reg) {
                    auto halfsel = [&](Id v, uint32_t sel) -> Id {
                        if (sel == 1) return b_.OpResult(SpvOpBitwiseAnd, t_u32_, {v, b_.ConstUint(0xFFFF)});
                        if (sel == 2) return b_.OpResult(SpvOpShiftRightLogical, t_u32_, {v, b_.ConstUint(16)});
                        return v;
                    };
                    a = halfsel(a, uint32_t(Bits(i, 35, 2)));
                    bv = halfsel(bv, uint32_t(Bits(i, 33, 2)));
                    c = halfsel(c, uint32_t(Bits(i, 31, 2)));
                    shift = uint32_t(Bits(i, 37, 2));
                }
                auto neg = [&](Id v) { return BitcastU(b_.OpResult(SpvOpSNegate, t_i32_, {BitcastI(v)})); };
                if (Bits(i, 51, 1)) a = neg(a);
                if (Bits(i, 50, 1)) bv = neg(bv);
                if (Bits(i, 49, 1)) c = neg(c);
                Id s = b_.OpResult(SpvOpIAdd, t_u32_, {a, bv});
                if (shift == 1) s = b_.OpResult(SpvOpShiftRightLogical, t_u32_, {s, b_.ConstUint(16)});
                if (shift == 2) s = b_.OpResult(SpvOpShiftLeftLogical, t_u32_, {s, b_.ConstUint(16)});
                SetU(dest, b_.OpResult(SpvOpIAdd, t_u32_, {s, c}));
                break;
            }
            case Op::ISCADD: {
                Id a = GetU(ra), bv = SrcBU(i, f, false);
                if (Bits(i, 49, 1)) a = BitcastU(b_.OpResult(SpvOpSNegate, t_i32_, {BitcastI(a)}));
                if (Bits(i, 48, 1)) bv = BitcastU(b_.OpResult(SpvOpSNegate, t_i32_, {BitcastI(bv)}));
                Id sh = b_.OpResult(SpvOpShiftLeftLogical, t_u32_, {a, b_.ConstUint(uint32_t(Bits(i, 39, 5)))});
                SetU(dest, b_.OpResult(SpvOpIAdd, t_u32_, {sh, bv}));
                break;
            }
            case Op::SHL: {
                Id s = b_.OpResult(SpvOpBitwiseAnd, t_u32_, {SrcBU(i, f, false), b_.ConstUint(31)});
                SetU(dest, b_.OpResult(SpvOpShiftLeftLogical, t_u32_, {GetU(ra), s}));
                break;
            }
            case Op::SHR: {
                Id s = b_.OpResult(SpvOpBitwiseAnd, t_u32_, {SrcBU(i, f, false), b_.ConstUint(31)});
                if (Bits(i, 48, 1))
                    SetU(dest, BitcastU(b_.OpResult(SpvOpShiftRightArithmetic, t_i32_, {BitcastI(GetU(ra)), s})));
                else
                    SetU(dest, b_.OpResult(SpvOpShiftRightLogical, t_u32_, {GetU(ra), s}));
                break;
            }
            case Op::LOP32I: {
                Id a = GetU(ra), bv = b_.ConstUint(uint32_t(Bits(i, 20, 32)));
                if (Bits(i, 55, 1)) a = b_.OpResult(SpvOpNot, t_u32_, {a});
                if (Bits(i, 56, 1)) bv = b_.OpResult(SpvOpNot, t_u32_, {bv});
                Id r;
                switch (Bits(i, 53, 2)) {
                    case 0: r = b_.OpResult(SpvOpBitwiseAnd, t_u32_, {a, bv}); break;
                    case 1: r = b_.OpResult(SpvOpBitwiseOr, t_u32_, {a, bv}); break;
                    case 2: r = b_.OpResult(SpvOpBitwiseXor, t_u32_, {a, bv}); break;
                    default: r = bv; break;
                }
                SetU(dest, r);
                break;
            }
            case Op::LOP: {
                Id a = GetU(ra), bv = SrcBU(i, f, false);
                if (Bits(i, 39, 1)) a = b_.OpResult(SpvOpNot, t_u32_, {a});
                if (Bits(i, 40, 1)) bv = b_.OpResult(SpvOpNot, t_u32_, {bv});
                Id r;
                switch (Bits(i, 41, 2)) {
                    case 0: r = b_.OpResult(SpvOpBitwiseAnd, t_u32_, {a, bv}); break;
                    case 1: r = b_.OpResult(SpvOpBitwiseOr, t_u32_, {a, bv}); break;
                    case 2: r = b_.OpResult(SpvOpBitwiseXor, t_u32_, {a, bv}); break;
                    default: r = bv; break;
                }
                SetU(dest, r);
                break;
            }
            default:
                Error(std::string("unsupported instruction ") + OpName(d.op));
                break;
        }
    }

    const uint8_t* code_;
    size_t size_;
    TranslateOptions opt_;
    Stage stage_ = Stage::Vertex;
    ShaderInfo info_;
    std::string error_;
    uint32_t pc_ = 0;
    bool predicated_ = false;

    spv::Builder b_;
    Id main_ = 0, first_label_ = 0;
    Id t_void_ = 0, t_bool_ = 0, t_f32_ = 0, t_u32_ = 0, t_i32_ = 0, t_v4f_ = 0, t_v4u_ = 0, t_v4i_ = 0, t_v2f_ = 0,
       t_v3f_ = 0, t_v2i_ = 0, t_v3i_ = 0, p_fn_u32_ = 0, p_fn_bool_ = 0;
    Id reg_[256] = {}, pred_[8] = {}, cbuf_[kMaxCbufs] = {};
    Id in_[32] = {}, in_type_[32] = {}, out_[32] = {};
    Id tex_[32] = {}, tex_type_[32] = {}, shadow_[32] = {}, shadow_type_[32] = {};
    Id tex_image_type_[32] = {}, shadow_image_type_[32] = {};
    Id image_[8] = {}, image_type_[8] = {};
    Id push_ = 0;
    Id position_ = 0, point_size_ = 0, frag_coord_ = 0, frag_depth_ = 0, point_coord_ = 0, vertex_index_ = 0,
       instance_index_ = 0;
    std::vector<Id> interface_;
    // Structured control flow from SSY/SYNC.
    struct Block {
        Id merge;
        uint32_t end;   // byte offset of the reconvergence point
        bool closed;    // branch to merge already emitted
        std::vector<Known> known_before;  // register knowledge at the block's start
    };
    std::vector<Block> blocks_;
    std::vector<uint32_t> ssy_;
    uint32_t skip_until_ = 0;
    Id local_ = 0;  // local memory (STL/LDL)
    Known known_[256];
};

}  // namespace

bool GetStage(const uint8_t* code, size_t size, Stage* stage) {
    if (size < kCodeOffset) return false;
    uint32_t sph0;
    memcpy(&sph0, code + kSphOffset, 4);
    switch ((sph0 >> 10) & 0xF) {
        case 1: *stage = Stage::Vertex; return true;
        case 5: *stage = Stage::Fragment; return true;
        default: return false;
    }
}

TranslateResult Translate(const uint8_t* code, size_t size, const TranslateOptions& options) {
    return Translator(code, size, options).Run();
}

}  // namespace dsr::hle::nvn::shader
