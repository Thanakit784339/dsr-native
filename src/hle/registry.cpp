#include "hle/registry.h"

#include <mutex>
#include <unordered_map>

#include "common/log.h"

namespace dsr::hle {

namespace {
struct Table {
    std::vector<Entry> entries;
    std::unordered_map<std::string, size_t> by_name;
};
Table& GetTable() {
    static Table t;  // constructed on first use; safe during static initialization of other TUs
    return t;
}
}  // namespace

void Register(const char* mangled, void* ptr, bool is_data) {
    Table& t = GetTable();
    auto [it, inserted] = t.by_name.emplace(mangled, t.entries.size());
    if (!inserted) {
        LOGW("hle", "duplicate registration for %s (later one wins)", mangled);
        t.entries[it->second] = {mangled, ptr, is_data};
        return;
    }
    t.entries.push_back({mangled, ptr, is_data});
}

static const Entry* Find(const std::string& name) {
    Table& t = GetTable();
    auto it = t.by_name.find(name);
    return it == t.by_name.end() ? nullptr : &t.entries[it->second];
}

void* FindFunction(const std::string& name) {
    const Entry* e = Find(name);
    return e && !e->is_data ? e->ptr : nullptr;
}

void* FindData(const std::string& name) {
    const Entry* e = Find(name);
    return e && e->is_data ? e->ptr : nullptr;
}

const std::vector<Entry>& AllEntries() { return GetTable().entries; }

}  // namespace dsr::hle
