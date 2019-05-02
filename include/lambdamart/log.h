/*
 * Logging instruments.
 * Modified from LightGBM source code.
 */
#ifndef LAMBDAMART_LOG_H
#define LAMBDAMART_LOG_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstdarg>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <chrono>

namespace LambdaMART {

#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL thread_local
#endif

#ifndef CHECK
#define CHECK(condition)                                   \
  if (!(condition)) Log::Fatal("Check failed: " #condition \
     " at %s, line %d .\n", __FILE__,  __LINE__);
#endif

#ifndef CHECK_NOTNULL
#define CHECK_NOTNULL(pointer)                             \
  if ((pointer) == nullptr) Log::Fatal(#pointer " Can't be NULL at %s, line %d .\n", __FILE__,  __LINE__);
#endif


    enum class LogLevel: int {
        Fatal = -1,
        Warning = 0,
        Info = 1,
        Debug = 2,
        Trace = 3,
    };


/*!
* \brief A static Log class
*/
    class Log {
    public:
        /*!
        * \brief Resets the minimal log level. It is INFO by default.
        * \param level The new minimal log level.
        */
        static void ResetLogLevel(LogLevel level) {
            GetLevel() = level;
        }

        static void Trace(const char *format, ...) {
            va_list val;
            va_start(val, format);
            Write(LogLevel::Trace, GetCurrentTime(), "Trace", format, val);
            va_end(val);
        }
        static void Debug(const char *format, ...) {
            va_list val;
            va_start(val, format);
            Write(LogLevel::Debug, GetCurrentTime(), "Debug", format, val);
            va_end(val);
        }
        static void Info(const char *format, ...) {
            va_list val;
            va_start(val, format);
            Write(LogLevel::Info, GetCurrentTime(), "Info", format, val);
            va_end(val);
        }
        static void Warning(const char *format, ...) {
            va_list val;
            va_start(val, format);
            Write(LogLevel::Warning, GetCurrentTime(), "Warning", format, val);
            va_end(val);
        }
        static void Fatal(const char *format, ...) {
            va_list val;
            char str_buf[1024];
            va_start(val, format);
#ifdef _MSC_VER
            vsprintf_s(str_buf, format, val);
#else
            vsprintf(str_buf, format, val);
#endif
            va_end(val);
            fprintf(stderr, "[%ld] [Fatal] %s\n", GetCurrentTime(), str_buf);
            fflush(stderr);
            throw std::runtime_error(std::string(str_buf));
        }

    private:
        static long start_time;

        static void Write(LogLevel level, long cur_time, const char* level_str, const char *format, va_list val) {
            if (level <= GetLevel()) {  // omit the message with low level
                // write to STDOUT
                printf("[%.3lfs] [%s] ", double(cur_time) / 1000, level_str);
                vprintf(format, val);
                printf("\n");
                fflush(stdout);
            }
        }

        // a trick to use static variable in header file.
        // May be not good, but avoid to use an additional cpp file
        static LogLevel& GetLevel() { static THREAD_LOCAL LogLevel level = LogLevel::Info; return level; }
        static auto& GetStartTime() { static THREAD_LOCAL auto start_time = std::chrono::steady_clock::now(); return start_time; }
        static long GetCurrentTime() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - GetStartTime()).count(); }
    };

}
#endif