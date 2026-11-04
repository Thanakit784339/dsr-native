#pragma once
#include <cstdarg>
#include <cstdio>

namespace dsr {

enum class LogLevel { Trace, Debug, Info, Warn, Error, Fatal };

void LogV(LogLevel lvl, const char* tag, const char* fmt, va_list ap);
void Log(LogLevel lvl, const char* tag, const char* fmt, ...) __attribute__((format(printf, 3, 4)));
[[noreturn]] void Fatal(const char* tag, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

void SetLogLevel(LogLevel lvl);

// Mirror every log line into a file (truncates it). Each line is written immediately (no buffering).
bool OpenLogFile(const char* path);
// File descriptor of the log file, or -1 (used by the crash handler, which must not allocate).
int LogFileFd();

}  // namespace dsr

#define LOGT(tag, ...) ::dsr::Log(::dsr::LogLevel::Trace, tag, __VA_ARGS__)
#define LOGD(tag, ...) ::dsr::Log(::dsr::LogLevel::Debug, tag, __VA_ARGS__)
#define LOGI(tag, ...) ::dsr::Log(::dsr::LogLevel::Info, tag, __VA_ARGS__)
#define LOGW(tag, ...) ::dsr::Log(::dsr::LogLevel::Warn, tag, __VA_ARGS__)
#define LOGE(tag, ...) ::dsr::Log(::dsr::LogLevel::Error, tag, __VA_ARGS__)
#define FATAL(tag, ...) ::dsr::Fatal(tag, __VA_ARGS__)
