// Host tool: loads the game exactly like the Android runtime (parse, map, relocate, bind) and prints
// an import coverage report. On arm64 Linux it can also execute the game (--run) for HLE debugging.
//
//   dsr_dryrun <game_dir> [--run] [--list] [--verbose]

#include <cstdio>
#include <cstring>
#include <string>

#include "common/log.h"
#include "runtime/runtime.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <game_dir> [--run] [--list] [--verbose]\n", argv[0]);
        return 2;
    }
    dsr::RuntimeConfig cfg;
    cfg.game_dir = argv[1];
    cfg.save_dir = cfg.game_dir + "/../dsr_save";
    cfg.cache_dir = cfg.game_dir + "/../dsr_cache";
    cfg.dry_run = true;
    bool list = false;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--run")) cfg.dry_run = false;
        if (!strcmp(argv[i], "--list")) list = true;
        if (!strcmp(argv[i], "--verbose")) dsr::SetLogLevel(dsr::LogLevel::Debug);
    }
    std::string err;
    if (!dsr::LoadGame(cfg, &err)) {
        LOGE("main", "load failed: %s", err.c_str());
        return 1;
    }
    dsr::LogImportReport(list);
    return dsr::RunGame();
}
