#pragma once
#include <iostream>
#include <string>
#include <sstream>

#ifdef __linux__
using __int64 = long long;
#endif

class Measurements
{
    double  _freq_d = 0.0;
    __int64 _freq_i = 0;
    __int64 _start = 0;
public:
    Measurements();
    __int64  start();
    __int64  elapsed_ticks(__int64 tm_start=0);
    double   elapsed_sec  (__int64 tm_start=0)
    {
      return double(elapsed_ticks(tm_start)) / _freq_d;
    }
    std::string elapsed (__int64 tm_start=0);
};
inline std::string Measurements::elapsed(__int64 tm_start)
{
    std::ostringstream ss;
    const auto ticks = elapsed_ticks(tm_start);
    auto sec = ticks / _freq_i;
    if (auto hrs = sec / 3600; hrs > 0) {
        ss << hrs << " hrs ";
        sec -= hrs * 3600;
    }
    
    if (auto min = sec / 60; min > 0) {
        ss << min << " min ";
        sec -= min * 60;
    }
    if (sec > 0) {
        ss << sec << " s ";
    }
    auto usec = 1000000 * (ticks % _freq_i) / _freq_i;
    
    if (auto msec = usec / 1000; msec > 0) {
        ss << msec << " ms ";
        usec -= msec * 1000;
    }
    ss << usec << " us";
    return ss.str();
}
#ifdef _WIN64
#include <windows.h>
    
inline Measurements::Measurements()
{
    LARGE_INTEGER li;
    if (!QueryPerformanceFrequency(&li))
        std::cout << "QueryPerformanceFrequency failed!\n";
    else {
        _freq_i = li.QuadPart;
        _freq_d = double(li.QuadPart) / 1000.0;
    }
}
inline __int64 Measurements::start()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    _start = li.QuadPart;
    return li.QuadPart;
}
    
inline __int64 Measurements::elapsed_ticks(__int64 tm_start)
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    if (0 == tm_start) tm_start = _start;
    return li.QuadPart - tm_start;
}

#endif
#ifdef __linux__
#include <sys/time.h>
inline Measurements::Measurements()
{
    _freq_d = 1e6;
    _freq_i = 1000000;
}
inline __int64 Measurements::start()
{
    /*timespan ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    _start = ts.tv_sec * _freq_i + ts.tv_nsec / 1000;*/
    timeval ts;
    gettimeofday(&ts, NULL);
    _start = ts.tv_sec * _freq_i + ts.tv_usec;
    return _start;
}    
inline __int64 Measurements::elapsed_ticks(__int64 tm_start)
{
    /*timespan ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    if (0 == tm_start) {
    	tm_start = _start;
    }
    const __int64 tm_end = ts.tv_sec * _freq_i + ts.tv_nsec / 1000;
    return tm_end - tm_start;*/
    timeval te;
    gettimeofday(&te,NULL);
    if (0 == tm_start) {
    	tm_start = _start;
    }
    const __int64 tm_end = te.tv_sec * _freq_i + te.tv_usec;
    return tm_end - tm_start;
}
#endif



