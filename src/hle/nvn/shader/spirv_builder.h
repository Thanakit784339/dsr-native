#pragma once
// A small SPIR-V module builder: just enough for the Maxwell translator. Types and constants are
// deduplicated; instructions are appended to the section they belong to and stitched together in the
// order the SPIR-V spec requires.

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

#include "spirv.h"

namespace dsr::spv {

using Id = uint32_t;

class Builder {
public:
    // GLSL.std.450 opcodes used by the translator (avoids depending on the enum header's C++ form).
    enum GLSLstd450Shim : uint32_t {
        kRound = 1, kRoundEven = 2, kTrunc = 3, kFAbs = 4, kFloor = 8, kCeil = 9, kFract = 10, kSin = 13,
        kCos = 14, kExp2 = 29, kLog2 = 30, kSqrt = 31, kInverseSqrt = 32, kFMin = 37, kUMin = 38, kSMin = 39,
        kSAbsShim = 5, kFMax = 40, kUMax = 41, kSMax = 42, kFClamp = 43, kUClamp = 44, kSClamp = 45, kFma = 50,
    };

    Id NewId() { return next_id_++; }

    // ---- sections -----------------------------------------------------------------------------------------
    void Capability(SpvCapability c) {
        if (caps_.insert({c, 0}).second) Emit(capabilities_, SpvOpCapability, {uint32_t(c)});
    }
    Id ImportGlsl() {
        if (!glsl_) {
            glsl_ = NewId();
            std::vector<uint32_t> w = {glsl_};
            AppendString(w, "GLSL.std.450");
            Emit(imports_, SpvOpExtInstImport, w);
        }
        return glsl_;
    }
    void EntryPoint(SpvExecutionModel model, Id fn, const char* name, const std::vector<Id>& interface) {
        std::vector<uint32_t> w = {uint32_t(model), fn};
        AppendString(w, name);
        w.insert(w.end(), interface.begin(), interface.end());
        Emit(entry_points_, SpvOpEntryPoint, w);
    }
    void ExecutionMode(Id fn, SpvExecutionMode mode, std::initializer_list<uint32_t> args = {}) {
        std::vector<uint32_t> w = {fn, uint32_t(mode)};
        w.insert(w.end(), args.begin(), args.end());
        Emit(exec_modes_, SpvOpExecutionMode, w);
    }
    void Name(Id id, const char* name) {
        std::vector<uint32_t> w = {id};
        AppendString(w, name);
        Emit(debug_, SpvOpName, w);
    }
    void Decorate(Id id, SpvDecoration d, std::initializer_list<uint32_t> args = {}) {
        std::vector<uint32_t> w = {id, uint32_t(d)};
        w.insert(w.end(), args.begin(), args.end());
        Emit(annotations_, SpvOpDecorate, w);
    }
    void MemberDecorate(Id type, uint32_t member, SpvDecoration d, std::initializer_list<uint32_t> args = {}) {
        std::vector<uint32_t> w = {type, member, uint32_t(d)};
        w.insert(w.end(), args.begin(), args.end());
        Emit(annotations_, SpvOpMemberDecorate, w);
    }

    // ---- types ---------------------------------------------------------------------------------------------
    Id TypeVoid() { return Type(SpvOpTypeVoid, {}); }
    Id TypeBool() { return Type(SpvOpTypeBool, {}); }
    Id TypeFloat() { return Type(SpvOpTypeFloat, {32}); }
    Id TypeUint() { return Type(SpvOpTypeInt, {32, 0}); }
    Id TypeInt() { return Type(SpvOpTypeInt, {32, 1}); }
    Id TypeVector(Id comp, uint32_t n) { return Type(SpvOpTypeVector, {comp, n}); }
    Id TypePointer(SpvStorageClass sc, Id t) { return Type(SpvOpTypePointer, {uint32_t(sc), t}); }
    Id TypeFunction(Id ret) { return Type(SpvOpTypeFunction, {ret}); }
    Id TypeImage(Id sampled, SpvDim dim, uint32_t depth, uint32_t arrayed) {
        return Type(SpvOpTypeImage, {sampled, uint32_t(dim), depth, arrayed, 0, 1, uint32_t(SpvImageFormatUnknown)});
    }
    Id TypeStorageImage(Id sampled, SpvDim dim, uint32_t arrayed) {
        return Type(SpvOpTypeImage, {sampled, uint32_t(dim), 0, arrayed, 0, 2, uint32_t(SpvImageFormatUnknown)});
    }
    Id TypeSampledImage(Id image) { return Type(SpvOpTypeSampledImage, {image}); }
    // Arrays and structs are not deduplicated (they carry decorations).
    Id TypeArray(Id elem, uint32_t length) {
        Id id = NewId();
        Emit(types_, SpvOpTypeArray, {id, elem, ConstUint(length)});
        return id;
    }
    Id TypeStruct(std::initializer_list<Id> members) {
        Id id = NewId();
        std::vector<uint32_t> w = {id};
        w.insert(w.end(), members.begin(), members.end());
        Emit(types_, SpvOpTypeStruct, w);
        return id;
    }

    // ---- constants -----------------------------------------------------------------------------------------
    Id ConstUint(uint32_t v) { return Const(TypeUint(), v); }
    Id ConstInt(int32_t v) { return Const(TypeInt(), uint32_t(v)); }
    Id ConstFloat(float f) {
        uint32_t v;
        memcpy(&v, &f, 4);
        return Const(TypeFloat(), v);
    }
    Id ConstFloatBits(uint32_t v) { return Const(TypeFloat(), v); }
    Id ConstBool(bool b) {
        Id& slot = b ? true_ : false_;
        if (!slot) {
            slot = NewId();
            Emit(types_, b ? SpvOpConstantTrue : SpvOpConstantFalse, {TypeBool(), slot});
        }
        return slot;
    }
    Id ConstComposite(Id type, std::initializer_list<Id> parts) {
        std::vector<uint32_t> key = {type};
        key.insert(key.end(), parts.begin(), parts.end());
        auto it = composites_.find(key);
        if (it != composites_.end()) return it->second;
        Id id = NewId();
        std::vector<uint32_t> w = {type, id};
        w.insert(w.end(), parts.begin(), parts.end());
        Emit(types_, SpvOpConstantComposite, w);
        composites_[key] = id;
        return id;
    }

    // ---- globals -------------------------------------------------------------------------------------------
    Id Variable(Id ptr_type, SpvStorageClass sc) {
        Id id = NewId();
        Emit(types_, SpvOpVariable, {ptr_type, id, uint32_t(sc)});
        return id;
    }

    // ---- function body -------------------------------------------------------------------------------------
    // Function-local variables must be at the start of the first block; they go to a separate list.
    Id LocalVariable(Id ptr_type, Id initializer = 0) {
        Id id = NewId();
        if (initializer)
            Emit(locals_, SpvOpVariable, {ptr_type, id, uint32_t(SpvStorageClassFunction), initializer});
        else
            Emit(locals_, SpvOpVariable, {ptr_type, id, uint32_t(SpvStorageClassFunction)});
        return id;
    }
    void Label(Id id) { Emit(body_, SpvOpLabel, {id}); }
    void Op(SpvOp op, std::initializer_list<uint32_t> w) { Emit(body_, op, std::vector<uint32_t>(w)); }
    void Op(SpvOp op, const std::vector<uint32_t>& w) { Emit(body_, op, w); }
    Id OpResult(SpvOp op, Id type, std::initializer_list<uint32_t> args) {
        Id id = NewId();
        std::vector<uint32_t> w = {type, id};
        w.insert(w.end(), args.begin(), args.end());
        Emit(body_, op, w);
        return id;
    }
    Id OpResult(SpvOp op, Id type, const std::vector<uint32_t>& args) {
        Id id = NewId();
        std::vector<uint32_t> w = {type, id};
        w.insert(w.end(), args.begin(), args.end());
        Emit(body_, op, w);
        return id;
    }
    Id Glsl(Id type, GLSLstd450Shim op, std::initializer_list<Id> args) {
        Id id = NewId();
        std::vector<uint32_t> w = {type, id, ImportGlsl(), uint32_t(op)};
        w.insert(w.end(), args.begin(), args.end());
        Emit(body_, SpvOpExtInst, w);
        return id;
    }

    // Assemble: function wrapper (void main()) around locals + body.
    std::vector<uint32_t> Finish(Id main_fn, Id first_label) {
        std::vector<uint32_t> out = {SpvMagicNumber, 0x00010000 /* SPIR-V 1.0 */, 0x00440053 /* generator */, 0, 0};
        auto add = [&](const std::vector<uint32_t>& s) { out.insert(out.end(), s.begin(), s.end()); };
        add(capabilities_);
        add(imports_);
        std::vector<uint32_t> mm;
        Emit(mm, SpvOpMemoryModel, {uint32_t(SpvAddressingModelLogical), uint32_t(SpvMemoryModelGLSL450)});
        add(mm);
        add(entry_points_);
        add(exec_modes_);
        add(debug_);
        add(annotations_);
        Id void_t = TypeVoid();
        Id fn_t = TypeFunction(void_t);
        add(types_);
        std::vector<uint32_t> fn;
        Emit(fn, SpvOpFunction, {void_t, main_fn, uint32_t(SpvFunctionControlMaskNone), fn_t});
        Emit(fn, SpvOpLabel, {first_label});
        fn.insert(fn.end(), locals_.begin(), locals_.end());
        fn.insert(fn.end(), body_.begin(), body_.end());
        Emit(fn, SpvOpFunctionEnd, {});
        add(fn);
        out[3] = next_id_;  // bound
        return out;
    }


private:
    static void Emit(std::vector<uint32_t>& s, SpvOp op, const std::vector<uint32_t>& w) {
        s.push_back((uint32_t(w.size() + 1) << 16) | uint32_t(op));
        s.insert(s.end(), w.begin(), w.end());
    }
    static void AppendString(std::vector<uint32_t>& w, const char* str) {
        size_t n = strlen(str) + 1;
        size_t words = (n + 3) / 4;
        size_t at = w.size();
        w.resize(at + words, 0);
        memcpy(&w[at], str, n);
    }
    Id Type(SpvOp op, std::initializer_list<uint32_t> args) {
        std::vector<uint32_t> key = {uint32_t(op)};
        key.insert(key.end(), args.begin(), args.end());
        auto it = types_map_.find(key);
        if (it != types_map_.end()) return it->second;
        Id id = NewId();
        std::vector<uint32_t> w = {id};
        w.insert(w.end(), args.begin(), args.end());
        Emit(types_, op, w);
        types_map_[key] = id;
        return id;
    }
    Id Const(Id type, uint32_t v) {
        auto key = std::make_pair(type, v);
        auto it = consts_.find(key);
        if (it != consts_.end()) return it->second;
        Id id = NewId();
        Emit(types_, SpvOpConstant, {type, id, v});
        consts_[key] = id;
        return id;
    }

    Id next_id_ = 1;
    Id glsl_ = 0, true_ = 0, false_ = 0;
    std::map<uint32_t, int> caps_;
    std::map<std::vector<uint32_t>, Id> types_map_, composites_;
    std::map<std::pair<Id, uint32_t>, Id> consts_;
    std::vector<uint32_t> capabilities_, imports_, entry_points_, exec_modes_, debug_, annotations_, types_, locals_,
        body_;
};

}  // namespace dsr::spv
