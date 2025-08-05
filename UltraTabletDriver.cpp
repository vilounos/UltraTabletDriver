#include <windows.h>
#include <setupapi.h>
#include <hidsdi.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <conio.h>
#include <string>
#include <sstream>
#include <atomic>
#include <deque>
#include <mutex>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <intrin.h>
#include <commctrl.h>
#include <gdiplus.h>

#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "hid.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "gdiplus.lib")

using namespace Gdiplus;

#define FORCE_INLINE __forceinline
#define RESTRICT __restrict
#ifdef _MSC_VER
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#else
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#define CACHE_LINE_SIZE 64
#define PREFETCH_DISTANCE 256

struct alignas(16) Vec2f {
    union {
        struct { float x, y; };
        __m128 vec;
        float data[4];
    };
    float _pad[2];

    Vec2f() : x(0.0f), y(0.0f), _pad{ 0.0f, 0.0f } {}
    Vec2f(float x_, float y_) : x(x_), y(y_), _pad{ 0.0f, 0.0f } {}

    FORCE_INLINE Vec2f(const __m128& v) { vec = v; }

    FORCE_INLINE void Prefetch() const {
        _mm_prefetch(reinterpret_cast<const char*>(this), _MM_HINT_T0);
    }

    FORCE_INLINE void PrefetchWrite() const {
        _mm_prefetch(reinterpret_cast<const char*>(this), _MM_HINT_T1);
    }
};

struct alignas(16) Vec4f {
    union {
        struct { float x, y, z, w; };
        __m128 vec;
    };

    Vec4f() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    Vec4f(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

struct alignas(16) Vec2i {
    union {
        struct { int x, y; };
        __m128i vec;
        int data[4];
    };
    int _pad[2];

    Vec2i() : x(0), y(0), _pad{ 0, 0 } {}
    Vec2i(int x_, int y_) : x(x_), y(y_), _pad{ 0, 0 } {}

    FORCE_INLINE Vec2i(const __m128i& v) { vec = v; }

    FORCE_INLINE void Prefetch() const {
        _mm_prefetch(reinterpret_cast<const char*>(this), _MM_HINT_T0);
    }

    FORCE_INLINE void PrefetchWrite() const {
        _mm_prefetch(reinterpret_cast<const char*>(this), _MM_HINT_T1);
    }
};

class SIMDMath {
private:
    static bool sseSupported;
    static bool sse2Supported;
    static bool avxSupported;

public:
    static void Initialize() {
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        sseSupported = (cpuInfo[3] & (1 << 25)) != 0;
        sse2Supported = (cpuInfo[3] & (1 << 26)) != 0;
        avxSupported = (cpuInfo[2] & (1 << 28)) != 0;
    }

    static bool HasSSE() { return sseSupported; }
    static bool HasSSE2() { return sse2Supported; }
    static bool HasAVX() { return avxSupported; }


    static bool HasSSE3() {
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        return (cpuInfo[2] & (1 << 0)) != 0;
    }

    static bool HasSSSE3() {
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        return (cpuInfo[2] & (1 << 9)) != 0;
    }

    static bool HasSSE41() {
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        return (cpuInfo[2] & (1 << 19)) != 0;
    }

    static bool HasSSE42() {
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        return (cpuInfo[2] & (1 << 20)) != 0;
    }

    FORCE_INLINE static Vec2f Add(const Vec2f& a, const Vec2f& b) {
        Vec2f result;
        result.vec = _mm_add_ps(a.vec, b.vec);
        return result;
    }

    FORCE_INLINE static Vec2f Sub(const Vec2f& a, const Vec2f& b) {
        Vec2f result;
        result.vec = _mm_sub_ps(a.vec, b.vec);
        return result;
    }

    FORCE_INLINE static Vec2f Mul(const Vec2f& a, const Vec2f& b) {
        Vec2f result;
        result.vec = _mm_mul_ps(a.vec, b.vec);
        return result;
    }

    FORCE_INLINE static Vec2f Scale(const Vec2f& a, float scale) {
        Vec2f result;
        __m128 vscale = _mm_set1_ps(scale);
        result.vec = _mm_mul_ps(a.vec, vscale);
        return result;
    }

    FORCE_INLINE static float Dot(const Vec2f& a, const Vec2f& b) {
        if (HasSSE41()) {

            __m128 dp = _mm_dp_ps(a.vec, b.vec, 0x31);
            return _mm_cvtss_f32(dp);
        }
        else {
            __m128 mul = _mm_mul_ps(a.vec, b.vec);
            __m128 hadd = _mm_hadd_ps(mul, mul);
            return _mm_cvtss_f32(hadd);
        }
    }

    FORCE_INLINE static float Length(const Vec2f& a) {

        if (HasSSE41()) {
            __m128 dp = _mm_dp_ps(a.vec, a.vec, 0x31);
            return _mm_cvtss_f32(_mm_sqrt_ss(dp));
        }
        else {
            __m128 mul = _mm_mul_ps(a.vec, a.vec);
            __m128 hadd = _mm_hadd_ps(mul, mul);
            __m128 sqrt_result = _mm_sqrt_ss(hadd);
            return _mm_cvtss_f32(sqrt_result);
        }
    }

    FORCE_INLINE static float LengthSquared(const Vec2f& a) {
        if (HasSSE41()) {
            __m128 dp = _mm_dp_ps(a.vec, a.vec, 0x31);
            return _mm_cvtss_f32(dp);
        }
        else {
            __m128 mul = _mm_mul_ps(a.vec, a.vec);
            __m128 hadd = _mm_hadd_ps(mul, mul);
            return _mm_cvtss_f32(hadd);
        }
    }

    FORCE_INLINE static Vec2f Lerp(const Vec2f& a, const Vec2f& b, float t) {
        Vec2f result;
        __m128 vt = _mm_set1_ps(t);
        __m128 diff = _mm_sub_ps(b.vec, a.vec);
        __m128 lerp = _mm_mul_ps(diff, vt);
        result.vec = _mm_add_ps(a.vec, lerp);
        return result;
    }

    FORCE_INLINE static Vec2f ApplyRotation(const Vec2f& point, float cosAngle, float sinAngle) {
        Vec2f result;
        result.x = point.x * cosAngle - point.y * sinAngle;
        result.y = point.x * sinAngle + point.y * cosAngle;
        return result;
    }

    FORCE_INLINE static Vec2f NormalizeCoords(const Vec2i& raw, const Vec2i& areaMin, const Vec2i& areaSize) {
        __m128i diff = _mm_sub_epi32(raw.vec, areaMin.vec);
        __m128 fdiff = _mm_cvtepi32_ps(diff);
        __m128 fsize = _mm_cvtepi32_ps(areaSize.vec);
        __m128 normalized = _mm_div_ps(fdiff, fsize);

        Vec2f result;
        result.vec = normalized;
        return result;
    }


    FORCE_INLINE static void NormalizeCoordsBatch(const Vec2i* RESTRICT raw, const Vec2i& areaMin, const Vec2i& areaSize, Vec2f* RESTRICT results, int count) {
        const __m128i minVec = areaMin.vec;
        const __m128 sizeVec = _mm_cvtepi32_ps(areaSize.vec);

        for (int i = 0; i < count; i++) {
            __m128i diff = _mm_sub_epi32(raw[i].vec, minVec);
            __m128 fdiff = _mm_cvtepi32_ps(diff);
            results[i].vec = _mm_div_ps(fdiff, sizeVec);
        }
    }

    FORCE_INLINE static Vec2i MapToScreen(const Vec2f& normalized, const Vec2i& screenSize, const Vec2i& screenOffset) {
        __m128 screenSizeF = _mm_cvtepi32_ps(screenSize.vec);
        __m128 scaled = _mm_mul_ps(normalized.vec, screenSizeF);
        __m128i scaledI = _mm_cvtps_epi32(scaled);
        __m128i result = _mm_add_epi32(scaledI, screenOffset.vec);

        Vec2i resultVec;
        resultVec.vec = result;
        return resultVec;
    }


    FORCE_INLINE static Vec2i Clamp(const Vec2i& value, const Vec2i& minVal, const Vec2i& maxVal) {
        Vec2i result;
        result.vec = _mm_max_epi32(_mm_min_epi32(value.vec, maxVal.vec), minVal.vec);
        return result;
    }


    FORCE_INLINE static Vec2f Clamp(const Vec2f& value, const Vec2f& minVal, const Vec2f& maxVal) {
        Vec2f result;
        result.vec = _mm_max_ps(_mm_min_ps(value.vec, maxVal.vec), minVal.vec);
        return result;
    }
};

bool SIMDMath::sseSupported = false;
bool SIMDMath::sse2Supported = false;
bool SIMDMath::avxSupported = false;

enum class TabletType {
    UNKNOWN = 0,
    WACOM_CTL672 = 1,
    XPPEN_STAR_G640 = 2,
    WACOM_CTL472 = 3
};

struct TabletSpec {
    int maxX;
    int maxY;
    int widthMM;
    int heightMM;
    TabletType type;
    std::string name;
};

struct Monitor {
    int x, y, width, height;
    std::string name;
    bool isPrimary;
};

template<typename T, size_t Size>
class alignas(64) LockFreeRingBuffer {
private:
    static_assert((Size& (Size - 1)) == 0, "Size must be power of 2");

    alignas(64) T buffer[Size];
    alignas(64) std::atomic<uint64_t> writeIndex;
    alignas(64) std::atomic<uint64_t> readIndex;

    static constexpr uint64_t MASK = Size - 1;

public:
    LockFreeRingBuffer() : writeIndex(0), readIndex(0) {}

    FORCE_INLINE bool TryPush(const T& item) {
        const uint64_t currentWrite = writeIndex.load(std::memory_order_relaxed);
        const uint64_t nextWrite = currentWrite + 1;
        const uint64_t currentRead = readIndex.load(std::memory_order_acquire);


        if (UNLIKELY((nextWrite & MASK) == (currentRead & MASK))) {
            return false;
        }


        _mm_prefetch(reinterpret_cast<const char*>(&buffer[currentWrite & MASK]), _MM_HINT_T0);

        buffer[currentWrite & MASK] = item;


        std::atomic_thread_fence(std::memory_order_release);
        writeIndex.store(nextWrite, std::memory_order_release);

        return true;
    }

    FORCE_INLINE bool TryPop(T& item) {
        const uint64_t currentRead = readIndex.load(std::memory_order_relaxed);
        const uint64_t currentWrite = writeIndex.load(std::memory_order_acquire);


        if (UNLIKELY(currentRead == currentWrite)) {
            return false;
        }


        _mm_prefetch(reinterpret_cast<const char*>(&buffer[currentRead & MASK]), _MM_HINT_T0);

        item = buffer[currentRead & MASK];


        std::atomic_thread_fence(std::memory_order_acquire);
        readIndex.store(currentRead + 1, std::memory_order_release);

        return true;
    }

    FORCE_INLINE bool IsEmpty() const {
        return readIndex.load(std::memory_order_acquire) == writeIndex.load(std::memory_order_acquire);
    }

    FORCE_INLINE size_t Size() const {
        return (writeIndex.load(std::memory_order_acquire) - readIndex.load(std::memory_order_acquire)) & MASK;
    }
};

struct DriverConfig {
    int areaWidth = 28;
    int areaHeight = 22;
    int areaCenterX = 80;
    int areaCenterY = 80;
    int rotation = 0;
    bool movementPrediction = false;
    int predictionStrength = 8;
    int maxPredictionDistance = 100;
    bool clickEnabled = false;
    int currentMonitor = 0;
    bool smoothingEnabled = false;
    int smoothingStrength = 15;
    bool jitterReducerEnabled = false;
    int jitterReducerThreshold = 2;
};

struct alignas(16) TabletData {
    Vec2i rawPos;
    bool inProximity;
    bool isTouching;
    bool isValid;
    float _pad;
    std::chrono::high_resolution_clock::time_point timestamp;
};


class HighPerformanceTabletDriver;
typedef bool (HighPerformanceTabletDriver::* TabletParserFunc)(BYTE*, DWORD, TabletData&);

struct alignas(64) VelocityHistorySOA {
    static const int HISTORY_SIZE = 16;

    alignas(64) float velocityX[HISTORY_SIZE];
    alignas(64) float velocityY[HISTORY_SIZE];
    alignas(64) float weights[HISTORY_SIZE];
    alignas(64) uint64_t timestamps[HISTORY_SIZE];

    alignas(64) std::atomic<int> writeIndex{ 0 };
    alignas(64) std::atomic<int> count{ 0 };

    VelocityHistorySOA() {
        memset(velocityX, 0, sizeof(velocityX));
        memset(velocityY, 0, sizeof(velocityY));
        memset(weights, 0, sizeof(weights));
        memset(timestamps, 0, sizeof(timestamps));
    }

    FORCE_INLINE void AddSample(float vx, float vy, uint64_t timestamp) {
        int idx = writeIndex.load(std::memory_order_relaxed);


        _mm_prefetch(reinterpret_cast<const char*>(&velocityX[idx]), _MM_HINT_T0);


        velocityX[idx] = vx;
        velocityY[idx] = vy;
        timestamps[idx] = timestamp;
        weights[idx] = 1.0f;


        int nextIdx = (idx + 1) & (HISTORY_SIZE - 1);
        writeIndex.store(nextIdx, std::memory_order_release);

        int currentCount = count.load(std::memory_order_relaxed);
        if (LIKELY(currentCount < HISTORY_SIZE)) {
            count.store(currentCount + 1, std::memory_order_release);
        }
    }

    FORCE_INLINE Vec2f CalculateWeightedAverage() const {
        int sampleCount = count.load(std::memory_order_acquire);
        if (UNLIKELY(sampleCount == 0)) return Vec2f(0.0f, 0.0f);


        _mm_prefetch(reinterpret_cast<const char*>(velocityX), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(velocityY), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(weights), _MM_HINT_T0);


        if (sampleCount > 4) {
            _mm_prefetch(reinterpret_cast<const char*>(&velocityX[4]), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&velocityY[4]), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&weights[4]), _MM_HINT_T0);
        }

        __m128 sumX = _mm_setzero_ps();
        __m128 sumY = _mm_setzero_ps();
        __m128 totalWeight = _mm_setzero_ps();


        int simdCount = sampleCount & ~3;
        for (int i = 0; i < simdCount; i += 4) {
            __m128 vx = _mm_load_ps(&velocityX[i]);
            __m128 vy = _mm_load_ps(&velocityY[i]);
            __m128 w = _mm_load_ps(&weights[i]);


            sumX = _mm_add_ps(sumX, _mm_mul_ps(vx, w));
            sumY = _mm_add_ps(sumY, _mm_mul_ps(vy, w));
            totalWeight = _mm_add_ps(totalWeight, w);
        }


        for (int i = simdCount; i < sampleCount; i++) {
            float vx = velocityX[i];
            float vy = velocityY[i];
            float w = weights[i];

            sumX = _mm_add_ss(sumX, _mm_set_ss(vx * w));
            sumY = _mm_add_ss(sumY, _mm_set_ss(vy * w));
            totalWeight = _mm_add_ss(totalWeight, _mm_set_ss(w));
        }


        sumX = _mm_hadd_ps(sumX, sumY);
        totalWeight = _mm_hadd_ps(totalWeight, totalWeight);

        sumX = _mm_hadd_ps(sumX, sumX);
        totalWeight = _mm_hadd_ps(totalWeight, totalWeight);

        float weightSum = _mm_cvtss_f32(totalWeight);
        if (LIKELY(weightSum > 0.0f)) {
            float invWeight = 1.0f / weightSum;
            return Vec2f(_mm_cvtss_f32(sumX) * invWeight,
                _mm_cvtss_f32(_mm_shuffle_ps(sumX, sumX, 1)) * invWeight);
        }

        return Vec2f(0.0f, 0.0f);
    }

    void Clear() {
        count.store(0, std::memory_order_release);
        writeIndex.store(0, std::memory_order_release);
    }
};

class HighPerformanceTabletDriver {
private:
    TabletParserFunc tabletParser;

    struct alignas(CACHE_LINE_SIZE) PreCalculatedConstants {

        float scaleX, scaleY;
        int maxX, maxY;
        int centerOffsetX, centerOffsetY;
        float cosRotation, sinRotation;
        bool needsRotation;


        float smoothingThreshold;
        float smoothingFactor;
        int maxJumpX, maxJumpY;
        float predictionTimeBase;
        float predictionScale;

    } constants;


    struct alignas(CACHE_LINE_SIZE) HotPathData {
        std::atomic<Vec2i> smoothedRawPos;
        std::atomic<bool> hasSmoothedData{ false };
        std::atomic<bool> fastDataValid{ false };
        std::atomic<bool> hasLastPosition{ false };
        std::atomic<bool> wasInProximityLast{ false };
        std::atomic<bool> lastTouchState{ false };
        std::atomic<bool> currentlyPressed{ false };

    } hotData;

    alignas(CACHE_LINE_SIZE) TabletData fastAccessData;

    static constexpr size_t TABLET_BUFFER_SIZE = 32;
    static constexpr size_t BATCH_BUFFER_SIZE = 8;

    LockFreeRingBuffer<TabletData, TABLET_BUFFER_SIZE> tabletDataBuffer;
    LockFreeRingBuffer<TabletData, BATCH_BUFFER_SIZE> batchBuffer;

    VelocityHistorySOA velocityHistory;

    DWORD_PTR systemAffinityMask;
    DWORD_PTR processAffinityMask;
    std::vector<int> availableCores;

    std::chrono::high_resolution_clock::time_point lastPositionTime;
    std::chrono::high_resolution_clock::time_point secondLastPositionTime;

    HWND mainWindow = nullptr;
    HWND statusLabel = nullptr;
    HWND visualArea = nullptr;
    std::thread guiThread;
    std::atomic<bool> guiRunning{ false };
    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;

    enum {
        ID_CENTER_UP = 1001, ID_CENTER_DOWN, ID_CENTER_LEFT, ID_CENTER_RIGHT,


        ID_AREA_WIDTH_DEC_1, ID_AREA_WIDTH_INC_1,
        ID_AREA_WIDTH_DEC_10, ID_AREA_WIDTH_INC_10,
        ID_AREA_WIDTH_DEC_50, ID_AREA_WIDTH_INC_50,
        ID_AREA_WIDTH_DEC_100, ID_AREA_WIDTH_INC_100,


        ID_AREA_HEIGHT_DEC_1, ID_AREA_HEIGHT_INC_1,
        ID_AREA_HEIGHT_DEC_10, ID_AREA_HEIGHT_INC_10,
        ID_AREA_HEIGHT_DEC_50, ID_AREA_HEIGHT_INC_50,
        ID_AREA_HEIGHT_DEC_100, ID_AREA_HEIGHT_INC_100,


        ID_CENTER_X_DEC_1, ID_CENTER_X_INC_1,
        ID_CENTER_X_DEC_10, ID_CENTER_X_INC_10,
        ID_CENTER_X_DEC_100, ID_CENTER_X_INC_100,


        ID_CENTER_Y_DEC_1, ID_CENTER_Y_INC_1,
        ID_CENTER_Y_DEC_10, ID_CENTER_Y_INC_10,
        ID_CENTER_Y_DEC_100, ID_CENTER_Y_INC_100,

        ID_ROTATION_TOGGLE, ID_MONITOR_SWITCH,
        ID_START_STOP,


        ID_PREDICTION_TOGGLE,
        ID_PREDICTION_STR_DEC_1, ID_PREDICTION_STR_INC_1,
        ID_PREDICTION_STR_DEC_10, ID_PREDICTION_STR_INC_10,


        ID_PREDICTION_DIST_DEC_1, ID_PREDICTION_DIST_INC_1,
        ID_PREDICTION_DIST_DEC_10, ID_PREDICTION_DIST_INC_10,
        ID_PREDICTION_DIST_DEC_100, ID_PREDICTION_DIST_INC_100,

        ID_CLICK_TOGGLE,


        ID_SMOOTHING_TOGGLE,
        ID_SMOOTHING_STR_DEC_1, ID_SMOOTHING_STR_INC_1,
        ID_SMOOTHING_STR_DEC_10, ID_SMOOTHING_STR_INC_10,


        ID_JITTER_REDUCER_TOGGLE,
        ID_JITTER_REDUCER_DEC_1, ID_JITTER_REDUCER_INC_1,
        ID_JITTER_REDUCER_DEC_10, ID_JITTER_REDUCER_INC_10,

        ID_VISUAL_AREA = 2000
    };

    HANDLE deviceHandle;
    std::atomic<bool> running{ false };
    std::atomic<bool> configMode{ false };
    TabletSpec currentTablet;
    DriverConfig config;

    std::thread inputThread;
    std::thread processingThread;
    std::thread configThread;
    std::thread loggingThread;

    std::vector<Monitor> monitors;
    int SCREEN_WIDTH = GetSystemMetrics(SM_CXSCREEN);
    int SCREEN_HEIGHT = GetSystemMetrics(SM_CYSCREEN);

    alignas(CACHE_LINE_SIZE) std::atomic<Vec2f> lastScreenPos;

    std::atomic<bool> emergencyShutdown{ false };

    struct alignas(CACHE_LINE_SIZE) ValidationData {
        std::atomic<Vec2i> lastValidRawPos;
        std::atomic<bool> hasValidData{ false };

    } validationData;

    struct alignas(CACHE_LINE_SIZE) CurrentPositionData {
        std::atomic<int> currentTabletX{ 0 };
        std::atomic<int> currentTabletY{ 0 };
        std::atomic<int> currentScreenX{ 0 };
        std::atomic<int> currentScreenY{ 0 };
        std::atomic<bool> isCurrentlyClicking{ false };
        std::atomic<int> lastJitterReducerX{ 0 };
        std::atomic<int> lastJitterReducerY{ 0 };
        std::atomic<bool> hasJitterReducerPos{ false };

    } currentPos;

    struct alignas(CACHE_LINE_SIZE) PredictionData {
        std::atomic<Vec2f> currentVelocity;
        std::atomic<double> predictionAmount{ 0.0 };

    } predictionData;

    const TabletSpec TABLET_SPECS[4] = {
        {0, 0, 0, 0, TabletType::UNKNOWN, "Unknown"},
        {21610, 13498, 216, 135, TabletType::WACOM_CTL672, "Wacom CTL-672"},
        {30480, 20320, 152, 102, TabletType::XPPEN_STAR_G640, "XPPen Star G 640"},
        {15200, 9500, 152, 95, TabletType::WACOM_CTL472, "Wacom CTL-472"}
    };

    FORCE_INLINE bool ParseWacomCTL672(BYTE* data, DWORD length, TabletData& output) {
        if (length < 8) return false;

        int rawX, rawY;
        bool inProximity, isTouching;

        if (data[0] == 0x01) {
            rawX = data[2] | (data[3] << 8);
            rawY = data[4] | (data[5] << 8);
            inProximity = (data[1] & 0x01) != 0;
            isTouching = (data[1] & 0x02) != 0;
        }
        else if (data[0] == 0x02) {
            rawX = data[2] | (data[3] << 8);
            rawY = data[4] | (data[5] << 8);
            inProximity = (data[1] & 0x20) != 0;
            isTouching = (data[1] & 0x01) != 0;
        }
        else {
            return false;
        }

        if (rawX == 0xFFFF || rawY == 0xFFFF || rawX == 0x7FFF || rawY == 0x7FFF) {
            return false;
        }

        output.rawPos = Vec2i(rawX, rawY);
        output.inProximity = inProximity;
        output.isTouching = isTouching;
        output.timestamp = std::chrono::high_resolution_clock::now();
        output.isValid = IsValidTabletDataFast(output.rawPos, inProximity);

        return output.isValid;
    }

    FORCE_INLINE bool ParseWacomCTL472(BYTE* data, DWORD length, TabletData& output) {
        if (length < 10) return false;

        int rawX, rawY;
        bool inProximity, isTouching;

        if (data[0] == 0x01 || data[0] == 0x02) {
            rawX = data[2] | (data[3] << 8);
            rawY = data[4] | (data[5] << 8);
            BYTE status = data[1];
            inProximity = (status & 0x20) != 0;
            isTouching = (status & 0x01) != 0;
        }
        else if (data[0] == 0x10 && length >= 8) {
            rawX = data[1] | (data[2] << 8);
            rawY = data[3] | (data[4] << 8);
            inProximity = (data[5] & 0x40) != 0;
            isTouching = (data[5] & 0x01) != 0;
        }
        else {
            return false;
        }

        if (rawX == 0xFFFF || rawY == 0xFFFF || rawX == 0x7FFF || rawY == 0x7FFF) {
            return false;
        }

        output.rawPos = Vec2i(rawX, rawY);
        output.inProximity = inProximity;
        output.isTouching = isTouching;
        output.timestamp = std::chrono::high_resolution_clock::now();
        output.isValid = IsValidTabletDataFast(output.rawPos, inProximity);

        return output.isValid;
    }

    FORCE_INLINE bool ParseXPPen(BYTE* data, DWORD length, TabletData& output) {
        if (length < 14) return false;

        if (data[0] != 0x02 || (data[1] != 0xA0 && data[1] != 0xA1)) {
            return false;
        }

        int rawX = data[2] | (data[3] << 8);
        int rawY = data[4] | (data[5] << 8);

        if (rawX == 0xFFFF || rawY == 0xFFFF || rawX == 0x7FFF || rawY == 0x7FFF) {
            return false;
        }

        output.rawPos = Vec2i(rawX, rawY);
        output.inProximity = true;
        output.isTouching = (data[1] == 0xA1);
        output.timestamp = std::chrono::high_resolution_clock::now();
        output.isValid = IsValidTabletDataFast(output.rawPos, true);

        return output.isValid;
    }

    FORCE_INLINE bool IsValidTabletDataFast(const Vec2i& rawPos, bool inProximity) {

        if (LIKELY(rawPos.x >= 0 && rawPos.y >= 0 &&
            rawPos.x <= constants.maxX && rawPos.y <= constants.maxY)) {


            if (UNLIKELY((rawPos.x == 0 && rawPos.y == 0) && !inProximity)) {
                return false;
            }


            validationData.lastValidRawPos.store(rawPos, std::memory_order_relaxed);
            validationData.hasValidData.store(true, std::memory_order_relaxed);
            return true;
        }

        return false;
    }

    FORCE_INLINE Vec2i ApplyRawDataSmoothing(const Vec2i& newRawPos, bool inProximity, bool isTouching) {

        if (LIKELY(!config.smoothingEnabled)) {
            return newRawPos;
        }

        if (UNLIKELY(!inProximity && !isTouching)) {
            hotData.hasSmoothedData.store(false, std::memory_order_relaxed);
            return newRawPos;
        }

        if (UNLIKELY(!hotData.hasSmoothedData.load(std::memory_order_relaxed))) {
            hotData.smoothedRawPos.store(newRawPos, std::memory_order_relaxed);
            hotData.hasSmoothedData.store(true, std::memory_order_relaxed);
            return newRawPos;
        }

        Vec2i currentSmoothed = hotData.smoothedRawPos.load(std::memory_order_relaxed);


        __m128i newVec = newRawPos.vec;
        __m128i currentVec = currentSmoothed.vec;
        __m128i deltaVec = _mm_sub_epi32(newVec, currentVec);
        __m128 deltaFloat = _mm_cvtepi32_ps(deltaVec);


        float distanceSquared = SIMDMath::LengthSquared(Vec2f(_mm_cvtss_f32(deltaFloat),
            _mm_cvtss_f32(_mm_shuffle_ps(deltaFloat, deltaFloat, 1))));


        const float thresholdSquared = constants.smoothingThreshold * constants.smoothingThreshold;
        if (LIKELY(distanceSquared < thresholdSquared)) {
            return currentSmoothed;
        }


        __m128 factor = _mm_set1_ps(constants.smoothingFactor);
        __m128 smoothed = _mm_add_ps(_mm_cvtepi32_ps(currentVec), _mm_mul_ps(deltaFloat, factor));
        __m128i smoothedInt = _mm_cvtps_epi32(smoothed);


        Vec2i minBounds(0, 0);
        Vec2i maxBounds(constants.maxX, constants.maxY);
        Vec2i result;
        result.vec = _mm_max_epi32(_mm_min_epi32(smoothedInt, maxBounds.vec), minBounds.vec);

        hotData.smoothedRawPos.store(result, std::memory_order_relaxed);
        return result;
    }

    void UpdatePreCalculatedConstants() {
        constants.maxX = currentTablet.maxX;
        constants.maxY = currentTablet.maxY;

        constants.maxJumpX = currentTablet.maxX;
        constants.maxJumpY = currentTablet.maxY;

        Vec2i areaMin, areaSize;
        CalculateAreaBounds(areaMin, areaSize);

        constants.scaleX = 1.0f / areaSize.x;
        constants.scaleY = 1.0f / areaSize.y;
        constants.centerOffsetX = areaMin.x;
        constants.centerOffsetY = areaMin.y;

        constants.smoothingThreshold = 2.0f;

        constants.smoothingFactor = 1.0f - (config.smoothingStrength * (0.04f / 3.0f));
        constants.smoothingFactor = max(0.1f, min(0.98f, constants.smoothingFactor));

        constants.predictionTimeBase = 0.012f;

        constants.predictionScale = config.predictionStrength * (0.1f / 4.0f);

        float angleRad = config.rotation * (3.14159265f / 2.0f);
        constants.cosRotation = cos(angleRad);
        constants.sinRotation = sin(angleRad);
        constants.needsRotation = (config.rotation != 0);
    }

    void InitializeTabletParser() {
        switch (currentTablet.type) {
        case TabletType::WACOM_CTL672:
            tabletParser = &HighPerformanceTabletDriver::ParseWacomCTL672;
            break;
        case TabletType::WACOM_CTL472:
            tabletParser = &HighPerformanceTabletDriver::ParseWacomCTL472;
            break;
        case TabletType::XPPEN_STAR_G640:
            tabletParser = &HighPerformanceTabletDriver::ParseXPPen;
            break;
        default:
            tabletParser = nullptr;
            break;
        }
    }

    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        HighPerformanceTabletDriver* driver = nullptr;

        if (uMsg == WM_NCCREATE) {
            CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
            driver = reinterpret_cast<HighPerformanceTabletDriver*>(pCreate->lpCreateParams);
            SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(driver));
        }
        else {
            driver = reinterpret_cast<HighPerformanceTabletDriver*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
        }

        if (driver) {
            return driver->HandleWindowMessage(hwnd, uMsg, wParam, lParam);
        }
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    static LRESULT CALLBACK VisualAreaProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        HighPerformanceTabletDriver* driver = reinterpret_cast<HighPerformanceTabletDriver*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

        if (uMsg == WM_PAINT && driver) {
            driver->DrawTabletArea(hwnd);
            return 0;
        }
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    LRESULT HandleWindowMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    void CreateControls(HWND hwnd);
    void DrawTabletArea(HWND hwnd);
    void HandleCommand(int commandId);
    void UpdateGUI();

    std::wstring GetRotationText();

    void InitializeCPUAffinity() {
        GetProcessAffinityMask(GetCurrentProcess(), &processAffinityMask, &systemAffinityMask);

        availableCores.clear();
        for (int i = 0; i < 64; i++) {
            if (processAffinityMask & (1ULL << i)) {
                availableCores.push_back(i);
            }
        }

        std::cout << "Available CPU cores: " << availableCores.size() << std::endl;

        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }

    void SetOptimalThreadAffinity(int threadType) {
        if (availableCores.size() < 2) return;

        DWORD_PTR affinityMask = 0;

        switch (threadType) {
        case 0:
            if (availableCores.size() > 0) {
                affinityMask = 1ULL << availableCores[0];
            }
            break;
        case 1:
            if (availableCores.size() > 1) {
                affinityMask = 1ULL << availableCores[1];
            }
            else if (availableCores.size() > 0) {
                affinityMask = 1ULL << availableCores[0];
            }
            break;
        case 2:
            if (availableCores.size() > 2) {
                affinityMask = 1ULL << availableCores[2];
            }
            else if (availableCores.size() > 0) {
                affinityMask = 1ULL << availableCores[availableCores.size() - 1];
            }
            break;
        }

        if (affinityMask != 0) {
            SetThreadAffinityMask(GetCurrentThread(), affinityMask);
        }

        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    }

    static BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
        std::vector<Monitor>* monitors = reinterpret_cast<std::vector<Monitor>*>(dwData);

        MONITORINFOEX monitorInfo;
        monitorInfo.cbSize = sizeof(MONITORINFOEX);

        if (GetMonitorInfo(hMonitor, &monitorInfo)) {
            Monitor monitor;
            monitor.x = monitorInfo.rcMonitor.left;
            monitor.y = monitorInfo.rcMonitor.top;
            monitor.width = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
            monitor.height = monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top;
            char deviceName[64];
            WideCharToMultiByte(CP_UTF8, 0, monitorInfo.szDevice, -1, deviceName, sizeof(deviceName), NULL, NULL);
            monitor.name = std::string(deviceName);
            monitor.isPrimary = (monitorInfo.dwFlags & MONITORINFOF_PRIMARY) != 0;

            monitors->push_back(monitor);
        }

        return TRUE;
    }

    void DetectMonitors() {
        monitors.clear();
        EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, reinterpret_cast<LPARAM>(&monitors));

        std::sort(monitors.begin(), monitors.end(), [](const Monitor& a, const Monitor& b) {
            if (a.isPrimary != b.isPrimary) return a.isPrimary;
            return a.x < b.x;
            });

        if (monitors.empty()) {
            Monitor fallback;
            fallback.x = 0;
            fallback.y = 0;
            fallback.width = SCREEN_WIDTH;
            fallback.height = SCREEN_HEIGHT;
            fallback.name = "Primary";
            fallback.isPrimary = true;
            monitors.push_back(fallback);
        }

        if (config.currentMonitor >= (int)monitors.size()) {
            config.currentMonitor = 0;
        }
    }

    void LoadConfig();
    void SaveConfig();

    TabletType DetectTabletType(USHORT vendorId, USHORT productId) {
        if (vendorId == 0x056A) {
            return PromptWacomTabletSelection();
        }
        else if (vendorId == 0x28BD || vendorId == 0x0483) {
            return TabletType::XPPEN_STAR_G640;
        }

        return TabletType::UNKNOWN;
    }

    TabletType PromptWacomTabletSelection() {
        std::cout << "\nDetected Wacom tablet. Please select your specific model:" << std::endl;
        std::cout << "1. Wacom CTL-672 (One by Wacom Medium)" << std::endl;
        std::cout << "2. Wacom CTL-472 (One by Wacom Small)" << std::endl;
        std::cout << "Enter your choice (1-2): ";

        int choice;
        while (!(std::cin >> choice) || choice < 1 || choice > 2) {
            std::cout << "Invalid input. Please enter 1 or 2: ";
            std::cin.clear();
            std::cin.ignore(10000, '\n');
        }

        switch (choice) {
        case 1:
            return TabletType::WACOM_CTL672;
        case 2:
            return TabletType::WACOM_CTL472;
        default:
            return TabletType::WACOM_CTL672;
        }
    }

    void SetDefaultArea() {
        switch (currentTablet.type) {
        case TabletType::WACOM_CTL672:
            config.areaWidth = 45;
            config.areaHeight = 58;
            config.areaCenterX = currentTablet.widthMM / 2;
            config.areaCenterY = currentTablet.heightMM / 2;
            break;
        case TabletType::WACOM_CTL472:
            config.areaWidth = 40;
            config.areaHeight = 25;
            config.areaCenterX = currentTablet.widthMM / 2;
            config.areaCenterY = currentTablet.heightMM / 2;
            break;
        case TabletType::XPPEN_STAR_G640:
            config.areaWidth = 28;
            config.areaHeight = 22;
            config.areaCenterX = 80;
            config.areaCenterY = 30;
            break;
        default:
            break;
        }
    }

    void OptimizeSystemForTablet() {
        timeBeginPeriod(1);

        SystemParametersInfo(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, 0, SPIF_SENDCHANGE);

        PROCESS_POWER_THROTTLING_STATE PowerThrottling{};
        PowerThrottling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        PowerThrottling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        PowerThrottling.StateMask = 0;

        SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
            &PowerThrottling, sizeof(PowerThrottling));
    }

    FORCE_INLINE void UpdateVelocityHistory(const Vec2f& newPosition, std::chrono::high_resolution_clock::time_point timestamp) {
        if (!hotData.hasLastPosition.load(std::memory_order_relaxed)) {
            lastPositionTime = timestamp;
            return;
        }

        Vec2f lastPos = lastScreenPos.load(std::memory_order_relaxed);

        auto timeDiff = std::chrono::duration_cast<std::chrono::microseconds>(timestamp - lastPositionTime);
        float deltaTime = timeDiff.count() / 1000000.0f;

        if (deltaTime > 0.0f && deltaTime < 0.1f) {
            float vx = (newPosition.x - lastPos.x) / deltaTime;
            float vy = (newPosition.y - lastPos.y) / deltaTime;

            uint64_t timestampMicros = std::chrono::duration_cast<std::chrono::microseconds>(
                timestamp.time_since_epoch()).count();

            velocityHistory.AddSample(vx, vy, timestampMicros);
        }

        secondLastPositionTime = lastPositionTime;
        lastPositionTime = timestamp;
    }

    FORCE_INLINE Vec2f CalculateSmoothedVelocity() {
        return velocityHistory.CalculateWeightedAverage();
    }

    FORCE_INLINE bool CalculateScreenPosition(const TabletData& data, Vec2f& screenPos) {

        data.rawPos.Prefetch();

        bool wasJustLifted = hotData.wasInProximityLast.load(std::memory_order_relaxed) && !data.inProximity;
        hotData.wasInProximityLast.store(data.inProximity, std::memory_order_relaxed);


        if (LIKELY(data.inProximity || data.isTouching)) {

        }
        else {
            hotData.hasLastPosition.store(false, std::memory_order_relaxed);
            velocityHistory.Clear();
            return false;
        }

        currentPos.currentTabletX.store(data.rawPos.x, std::memory_order_relaxed);
        currentPos.currentTabletY.store(data.rawPos.y, std::memory_order_relaxed);
        currentPos.isCurrentlyClicking.store(data.isTouching, std::memory_order_relaxed);


        const int areaMaxX = constants.centerOffsetX + (int)(1.0f / constants.scaleX);
        const int areaMaxY = constants.centerOffsetY + (int)(1.0f / constants.scaleY);

        bool outsideArea = UNLIKELY(data.rawPos.x < constants.centerOffsetX ||
            data.rawPos.y < constants.centerOffsetY ||
            data.rawPos.x > areaMaxX ||
            data.rawPos.y > areaMaxY);

        if (outsideArea) {

            if (UNLIKELY(config.clickEnabled && hotData.currentlyPressed.load(std::memory_order_relaxed))) {
                HandleMouseClick(false);
            }

        }


        Vec2i minBounds(constants.centerOffsetX, constants.centerOffsetY);
        Vec2i maxBounds(areaMaxX, areaMaxY);
        Vec2i clampedPos = SIMDMath::Clamp(data.rawPos, minBounds, maxBounds);


        __m128i offset = minBounds.vec;
        __m128i diff = _mm_sub_epi32(clampedPos.vec, offset);
        __m128 fdiff = _mm_cvtepi32_ps(diff);
        __m128 scale = _mm_set_ps(0.0f, 0.0f, constants.scaleY, constants.scaleX);
        __m128 normalized = _mm_mul_ps(fdiff, scale);

        float normalizedX = _mm_cvtss_f32(normalized);
        float normalizedY = _mm_cvtss_f32(_mm_shuffle_ps(normalized, normalized, _MM_SHUFFLE(1, 1, 1, 1)));

        if (constants.needsRotation) {
            float rotatedX, rotatedY;
            switch (config.rotation) {
            case 1:
                rotatedX = 1.0f - normalizedY;
                rotatedY = normalizedX;
                break;
            case 2:
                rotatedX = 1.0f - normalizedX;
                rotatedY = 1.0f - normalizedY;
                break;
            case 3:
                rotatedX = normalizedY;
                rotatedY = 1.0f - normalizedX;
                break;
            default:
                rotatedX = normalizedX;
                rotatedY = normalizedY;
                break;
            }
            normalizedX = rotatedX;
            normalizedY = rotatedY;
        }

        const Monitor& currentMonitor = monitors[config.currentMonitor];
        screenPos.x = currentMonitor.x + normalizedX * currentMonitor.width;
        screenPos.y = currentMonitor.y + normalizedY * currentMonitor.height;

        return true;
    }

    FORCE_INLINE void ApplyPrediction(const Vec2f& baseScreenPos, Vec2f& finalScreenPos,
        std::chrono::high_resolution_clock::time_point timestamp) {
        finalScreenPos = baseScreenPos;

        UpdateVelocityHistory(baseScreenPos, timestamp);


        if (LIKELY(config.movementPrediction && hotData.hasLastPosition.load(std::memory_order_relaxed))) {
            Vec2f smoothedVelocity = CalculateSmoothedVelocity();


            float velocityMagnitudeSquared = SIMDMath::LengthSquared(smoothedVelocity);
            const float minVelocitySquared = 1.0f * 1.0f;

            if (LIKELY(velocityMagnitudeSquared > minVelocitySquared)) {
                float velocityMagnitude = sqrt(velocityMagnitudeSquared);


                float adaptiveFactor = velocityMagnitude * 0.001f;
                adaptiveFactor = (adaptiveFactor > 1.0f) ? 1.0f : adaptiveFactor;

                float finalPredictionTime = constants.predictionTimeBase * adaptiveFactor * constants.predictionScale;

                Vec2f prediction = SIMDMath::Scale(smoothedVelocity, finalPredictionTime);


                float predictionDistanceSquared = SIMDMath::LengthSquared(prediction);
                const float maxDistanceSquared = config.maxPredictionDistance * config.maxPredictionDistance;

                if (UNLIKELY(predictionDistanceSquared > maxDistanceSquared)) {
                    float scale = config.maxPredictionDistance / sqrt(predictionDistanceSquared);
                    prediction = SIMDMath::Scale(prediction, scale);
                }

                finalScreenPos = SIMDMath::Add(baseScreenPos, prediction);

                predictionData.currentVelocity.store(smoothedVelocity, std::memory_order_relaxed);
                predictionData.predictionAmount.store(sqrt(predictionDistanceSquared), std::memory_order_relaxed);
            }
            else {

                static const Vec2f zeroVel(0.0f, 0.0f);
                predictionData.currentVelocity.store(zeroVel, std::memory_order_relaxed);
                predictionData.predictionAmount.store(0.0, std::memory_order_relaxed);
            }
        }
        else {

            static const Vec2f zeroVel(0.0f, 0.0f);
            predictionData.currentVelocity.store(zeroVel, std::memory_order_relaxed);
            predictionData.predictionAmount.store(0.0, std::memory_order_relaxed);
        }

        lastScreenPos.store(baseScreenPos, std::memory_order_relaxed);
        hotData.hasLastPosition.store(true, std::memory_order_relaxed);
    }


    FORCE_INLINE void MoveCursorToPosition(const Vec2f& screenPos) {
        int screenX = (int)screenPos.x;
        int screenY = (int)screenPos.y;

        const Monitor& currentMonitor = monitors[config.currentMonitor];


        screenX = max(currentMonitor.x, min(currentMonitor.x + currentMonitor.width - 1, screenX));
        screenY = max(currentMonitor.y, min(currentMonitor.y + currentMonitor.height - 1, screenY));


        if (LIKELY(config.jitterReducerEnabled)) {
            if (LIKELY(currentPos.hasJitterReducerPos.load(std::memory_order_relaxed))) {
                int lastX = currentPos.lastJitterReducerX.load(std::memory_order_relaxed);
                int lastY = currentPos.lastJitterReducerY.load(std::memory_order_relaxed);


                int deltaX = abs(screenX - lastX);
                int deltaY = abs(screenY - lastY);
                int maxDelta = max(deltaX, deltaY);


                if (UNLIKELY(maxDelta < config.jitterReducerThreshold)) {
                    return;
                }
            }


            currentPos.lastJitterReducerX.store(screenX, std::memory_order_relaxed);
            currentPos.lastJitterReducerY.store(screenY, std::memory_order_relaxed);
            currentPos.hasJitterReducerPos.store(true, std::memory_order_relaxed);
        }


        int normalizedX = (screenX * 65535) / (GetSystemMetrics(SM_CXSCREEN) - 1);
        int normalizedY = (screenY * 65535) / (GetSystemMetrics(SM_CYSCREEN) - 1);

        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
        input.mi.dx = normalizedX;
        input.mi.dy = normalizedY;

        SendInput(1, &input, sizeof(INPUT));

        currentPos.currentScreenX.store(screenX, std::memory_order_relaxed);
        currentPos.currentScreenY.store(screenY, std::memory_order_relaxed);
    }

    FORCE_INLINE void HandleMouseClick(bool isTouching) {
        if (!config.clickEnabled) return;

        bool lastTouch = hotData.lastTouchState.load(std::memory_order_relaxed);
        if (isTouching != lastTouch) {
            INPUT input = {};
            input.type = INPUT_MOUSE;

            if (isTouching && !hotData.currentlyPressed.load(std::memory_order_relaxed)) {
                input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
                SendInput(1, &input, sizeof(INPUT));
                hotData.currentlyPressed.store(true, std::memory_order_relaxed);
            }
            else if (!isTouching && hotData.currentlyPressed.load(std::memory_order_relaxed)) {
                input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
                SendInput(1, &input, sizeof(INPUT));
                hotData.currentlyPressed.store(false, std::memory_order_relaxed);
            }

            hotData.lastTouchState.store(isTouching, std::memory_order_relaxed);
        }
    }

    void ProcessRawTabletData(BYTE* buffer, DWORD length) {
        if (length < 6 || buffer == nullptr || tabletParser == nullptr) return;

        TabletData localData;

        if ((this->*tabletParser)(buffer, length, localData)) {
            Vec2i smoothedPos = ApplyRawDataSmoothing(localData.rawPos, localData.inProximity, localData.isTouching);
            localData.rawPos = smoothedPos;

            if (localData.inProximity || localData.isTouching) {
                fastAccessData = localData;
                _mm_sfence();
                hotData.fastDataValid = true;

                if (!batchBuffer.TryPush(localData)) {
                    ProcessBatchedReports();
                    tabletDataBuffer.TryPush(localData);
                }
            }
        }
    }

    FORCE_INLINE void ProcessBatchedReports() {
        TabletData batchData[BATCH_BUFFER_SIZE];
        size_t count = 0;

        while (count < BATCH_BUFFER_SIZE && batchBuffer.TryPop(batchData[count])) {
            count++;
        }

        for (size_t i = 0; i < count; i++) {
            if (batchData[i].isValid) {
                tabletDataBuffer.TryPush(batchData[i]);
            }
        }
    }

    void SafeInputLoop() {
        SetOptimalThreadAffinity(0);


        alignas(CACHE_LINE_SIZE) BYTE buffer[256];
        DWORD bytesRead;


        OVERLAPPED overlapped = {};
        overlapped.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

        while (LIKELY(running.load(std::memory_order_relaxed) && !emergencyShutdown.load(std::memory_order_relaxed))) {

            if (ReadFile(deviceHandle, buffer, sizeof(buffer), &bytesRead, &overlapped)) {

                ProcessRawTabletData(buffer, bytesRead);
            }
            else if (GetLastError() == ERROR_IO_PENDING) {

                DWORD waitResult = WaitForSingleObject(overlapped.hEvent, 1);
                if (waitResult == WAIT_OBJECT_0) {
                    if (GetOverlappedResult(deviceHandle, &overlapped, &bytesRead, FALSE)) {
                        ProcessRawTabletData(buffer, bytesRead);
                    }
                }
                ResetEvent(overlapped.hEvent);
            }
            else {

                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        CloseHandle(overlapped.hEvent);
    }

    void SafeProcessingLoop() {
        SetOptimalThreadAffinity(1);

        TabletData localData;
        Vec2f baseScreenPos, finalScreenPos;
        int idleCount = 0;

        while (LIKELY(running.load(std::memory_order_relaxed) && !emergencyShutdown.load(std::memory_order_relaxed))) {
            bool processedData = false;
            bool cursorMoved = false;


            if (LIKELY(hotData.fastDataValid.load(std::memory_order_acquire))) {
                localData = fastAccessData;
                std::atomic_thread_fence(std::memory_order_acquire);

                if (LIKELY(localData.isValid)) {
                    if (LIKELY(CalculateScreenPosition(localData, baseScreenPos))) {
                        ApplyPrediction(baseScreenPos, finalScreenPos, localData.timestamp);


                        MoveCursorToPosition(finalScreenPos);

                        HandleMouseClick(localData.isTouching);
                        processedData = true;
                        cursorMoved = true;
                    }
                }
                hotData.fastDataValid.store(false, std::memory_order_release);
            }


            if (UNLIKELY(!batchBuffer.IsEmpty())) {
                ProcessBatchedReports();
                processedData = true;
            }


            while (tabletDataBuffer.TryPop(localData) && !cursorMoved) {
                if (LIKELY(localData.isValid)) {
                    if (LIKELY(CalculateScreenPosition(localData, baseScreenPos))) {
                        ApplyPrediction(baseScreenPos, finalScreenPos, localData.timestamp);


                        MoveCursorToPosition(finalScreenPos);

                        HandleMouseClick(localData.isTouching);
                        processedData = true;
                        cursorMoved = true;
                    }
                }
            }


            if (cursorMoved) {
                TabletData discardData;
                while (tabletDataBuffer.TryPop(discardData)) {

                }
            }


            if (UNLIKELY(!processedData)) {
                idleCount++;
                if (idleCount < 10) {
                    _mm_pause();
                }
                else if (idleCount < 100) {
                    std::this_thread::yield();
                }
                else {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                    idleCount = 0;
                }
            }
            else {
                idleCount = 0;
            }
        }
    }


    void CalculateAreaBounds(Vec2i& areaMin, Vec2i& areaSize) const {


        Vec2i center(
            config.areaCenterX * currentTablet.maxX / currentTablet.widthMM,
            config.areaCenterY * currentTablet.maxY / currentTablet.heightMM
        );

        Vec2i halfSize(
            (config.areaWidth * currentTablet.maxX / currentTablet.widthMM) / 2,
            (config.areaHeight * currentTablet.maxY / currentTablet.heightMM) / 2
        );

        areaMin = Vec2i(center.x - halfSize.x, center.y - halfSize.y);
        areaSize = Vec2i(halfSize.x * 2, halfSize.y * 2);
    }

public:
    HighPerformanceTabletDriver() : deviceHandle(INVALID_HANDLE_VALUE), tabletParser(nullptr) {
        SIMDMath::Initialize();

        std::cout << "SIMD Support: SSE=" << (SIMDMath::HasSSE() ? "YES" : "NO")
            << ", SSE2=" << (SIMDMath::HasSSE2() ? "YES" : "NO")
            << ", SSE4.1=" << (SIMDMath::HasSSE41() ? "YES" : "NO")
            << ", AVX=" << (SIMDMath::HasAVX() ? "YES" : "NO") << std::endl;

        InitializeCPUAffinity();

        currentTablet = TABLET_SPECS[0];
        DetectMonitors();
        LoadConfig();

        Vec2f zero(0.0f, 0.0f);
        Vec2i zeroI(0, 0);
        predictionData.currentVelocity.store(zero);
        lastScreenPos.store(zero);
        validationData.lastValidRawPos.store(zeroI);

        hotData.fastDataValid = false;
        memset(&fastAccessData, 0, sizeof(TabletData));

        Vec2i zeroSmoothed(0, 0);
        hotData.smoothedRawPos.store(zeroSmoothed);
        hotData.hasSmoothedData.store(false);


        currentPos.hasJitterReducerPos.store(false);
        currentPos.lastJitterReducerX.store(0);
        currentPos.lastJitterReducerY.store(0);
    }

    ~HighPerformanceTabletDriver() {
        Stop();

        SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
        timeEndPeriod(1);

        if (deviceHandle != INVALID_HANDLE_VALUE) {
            CloseHandle(deviceHandle);
            deviceHandle = INVALID_HANDLE_VALUE;
        }
    }

    bool Initialize();
    void RunConfiguration();
    void WaitForExit();

    void Start() {
        if (deviceHandle == INVALID_HANDLE_VALUE) {
            std::cerr << "Device not initialized" << std::endl;
            return;
        }

        emergencyShutdown = false;
        running = true;
        hotData.hasLastPosition = false;
        hotData.wasInProximityLast = false;
        hotData.lastTouchState = false;
        hotData.currentlyPressed = false;

        validationData.hasValidData = false;
        Vec2i zeroPos(-1, -1);
        validationData.lastValidRawPos.store(zeroPos);

        hotData.hasSmoothedData.store(false);
        Vec2i zeroSmoothed(0, 0);
        hotData.smoothedRawPos.store(zeroSmoothed);


        currentPos.hasJitterReducerPos.store(false);
        currentPos.lastJitterReducerX.store(0);
        currentPos.lastJitterReducerY.store(0);

        velocityHistory.Clear();

        UpdatePreCalculatedConstants();

        inputThread = std::thread(&HighPerformanceTabletDriver::SafeInputLoop, this);
        processingThread = std::thread(&HighPerformanceTabletDriver::SafeProcessingLoop, this);
    }

    void Stop() {
        emergencyShutdown = true;
        running = false;
        guiRunning = false;

        if (hotData.currentlyPressed && config.clickEnabled) {
            INPUT input = {};
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
            hotData.currentlyPressed = false;
        }

        if (inputThread.joinable()) {
            inputThread.join();
        }
        if (processingThread.joinable()) {
            processingThread.join();
        }
        if (loggingThread.joinable()) {
            loggingThread.join();
        }
    }

    void MoveAreaCenter(int deltaX, int deltaY);
    void AdjustAreaSize(int widthDelta, int heightDelta);
    void ConfigureRotation();
    void SwitchMonitor();
    void CheckAndCorrectAreaBounds();
    void TogglePrediction();
    void AdjustPredictionStrength(int delta);
    void AdjustPredictionDistance(int delta);
    void ToggleSmoothing();
    void AdjustSmoothingStrength(int delta);
    void ToggleJitterReducer();
    void AdjustJitterReducer(int delta);
    void ToggleClick();
    void RestartDriver();
};

LRESULT HighPerformanceTabletDriver::HandleWindowMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        CreateControls(hwnd);
        return 0;

    case WM_COMMAND:
        HandleCommand(LOWORD(wParam));
        return 0;

    case WM_CLOSE:
        if (running) {
            Stop();
        }
        guiRunning = false;
        DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        GdiplusShutdown(gdiplusToken);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

void HighPerformanceTabletDriver::CreateControls(HWND hwnd) {

    HFONT normalFont = CreateFont(14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");

    HFONT bigFont = CreateFont(20, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");


    int windowWidth = 1000;
    int leftPanelX = 20;
    int centerX = windowWidth / 2;
    int rightPanelX = windowWidth - 280;


    WNDCLASS visualClass = {};
    visualClass.lpfnWndProc = VisualAreaProc;
    visualClass.hInstance = GetModuleHandle(nullptr);
    visualClass.lpszClassName = L"VisualArea";
    visualClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    RegisterClass(&visualClass);


    int visualWidth = 250;
    int visualHeight = 180;
    if (currentTablet.widthMM > 0 && currentTablet.heightMM > 0) {
        float tabletRatio = (float)currentTablet.heightMM / (float)currentTablet.widthMM;
        visualHeight = (int)(visualWidth * tabletRatio);
        if (visualHeight > 220) {
            visualHeight = 220;
            visualWidth = (int)(visualHeight / tabletRatio);
        }
    }

    visualArea = CreateWindow(L"VisualArea", L"", WS_VISIBLE | WS_CHILD | WS_BORDER,
        centerX - visualWidth / 2, 20, visualWidth, visualHeight, hwnd, (HMENU)ID_VISUAL_AREA, nullptr, nullptr);
    SetWindowLongPtr(visualArea, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));


    int startButtonY = 20 + visualHeight + 30;
    HWND startStopBtn = CreateWindow(L"BUTTON", running ? L"STOP" : L"START",
        WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
        centerX - 80, startButtonY, 160, 50, hwnd, (HMENU)ID_START_STOP, nullptr, nullptr);
    SendMessage(startStopBtn, WM_SETFONT, (WPARAM)bigFont, TRUE);


    int leftY = startButtonY + 20;


    CreateWindow(L"STATIC", L"Area X:", WS_VISIBLE | WS_CHILD,
        leftPanelX, leftY, 60, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"STATIC", (std::to_wstring(config.areaCenterX) + L" mm").c_str(),
        WS_VISIBLE | WS_CHILD | SS_CENTER,
        leftPanelX + 65, leftY, 60, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-100", WS_VISIBLE | WS_CHILD,
        leftPanelX + 130, leftY, 35, 20, hwnd, (HMENU)ID_CENTER_X_DEC_100, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-10", WS_VISIBLE | WS_CHILD,
        leftPanelX + 170, leftY, 30, 20, hwnd, (HMENU)ID_CENTER_X_DEC_10, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-1", WS_VISIBLE | WS_CHILD,
        leftPanelX + 205, leftY, 25, 20, hwnd, (HMENU)ID_CENTER_X_DEC_1, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+1", WS_VISIBLE | WS_CHILD,
        leftPanelX + 235, leftY, 25, 20, hwnd, (HMENU)ID_CENTER_X_INC_1, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+10", WS_VISIBLE | WS_CHILD,
        leftPanelX + 265, leftY, 30, 20, hwnd, (HMENU)ID_CENTER_X_INC_10, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+100", WS_VISIBLE | WS_CHILD,
        leftPanelX + 300, leftY, 35, 20, hwnd, (HMENU)ID_CENTER_X_INC_100, nullptr, nullptr);
    leftY += 25;


    CreateWindow(L"STATIC", L"Area Y:", WS_VISIBLE | WS_CHILD,
        leftPanelX, leftY, 60, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"STATIC", (std::to_wstring(config.areaCenterY) + L" mm").c_str(),
        WS_VISIBLE | WS_CHILD | SS_CENTER,
        leftPanelX + 65, leftY, 60, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-100", WS_VISIBLE | WS_CHILD,
        leftPanelX + 130, leftY, 35, 20, hwnd, (HMENU)ID_CENTER_Y_DEC_100, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-10", WS_VISIBLE | WS_CHILD,
        leftPanelX + 170, leftY, 30, 20, hwnd, (HMENU)ID_CENTER_Y_DEC_10, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-1", WS_VISIBLE | WS_CHILD,
        leftPanelX + 205, leftY, 25, 20, hwnd, (HMENU)ID_CENTER_Y_DEC_1, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+1", WS_VISIBLE | WS_CHILD,
        leftPanelX + 235, leftY, 25, 20, hwnd, (HMENU)ID_CENTER_Y_INC_1, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+10", WS_VISIBLE | WS_CHILD,
        leftPanelX + 265, leftY, 30, 20, hwnd, (HMENU)ID_CENTER_Y_INC_10, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+100", WS_VISIBLE | WS_CHILD,
        leftPanelX + 300, leftY, 35, 20, hwnd, (HMENU)ID_CENTER_Y_INC_100, nullptr, nullptr);
    leftY += 35;


    CreateWindow(L"STATIC", L"Area Height:", WS_VISIBLE | WS_CHILD,
        leftPanelX, leftY, 80, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"STATIC", (std::to_wstring(config.areaWidth) + L" mm").c_str(),
        WS_VISIBLE | WS_CHILD | SS_CENTER,
        leftPanelX + 85, leftY, 60, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-100", WS_VISIBLE | WS_CHILD,
        leftPanelX + 150, leftY, 35, 20, hwnd, (HMENU)ID_AREA_WIDTH_DEC_100, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-10", WS_VISIBLE | WS_CHILD,
        leftPanelX + 190, leftY, 30, 20, hwnd, (HMENU)ID_AREA_WIDTH_DEC_10, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-1", WS_VISIBLE | WS_CHILD,
        leftPanelX + 225, leftY, 25, 20, hwnd, (HMENU)ID_AREA_WIDTH_DEC_1, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+1", WS_VISIBLE | WS_CHILD,
        leftPanelX + 255, leftY, 25, 20, hwnd, (HMENU)ID_AREA_WIDTH_INC_1, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+10", WS_VISIBLE | WS_CHILD,
        leftPanelX + 285, leftY, 30, 20, hwnd, (HMENU)ID_AREA_WIDTH_INC_10, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+100", WS_VISIBLE | WS_CHILD,
        leftPanelX + 320, leftY, 35, 20, hwnd, (HMENU)ID_AREA_WIDTH_INC_100, nullptr, nullptr);
    leftY += 25;


    CreateWindow(L"STATIC", L"Area Width:", WS_VISIBLE | WS_CHILD,
        leftPanelX, leftY, 80, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"STATIC", (std::to_wstring(config.areaHeight) + L" mm").c_str(),
        WS_VISIBLE | WS_CHILD | SS_CENTER,
        leftPanelX + 85, leftY, 60, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-100", WS_VISIBLE | WS_CHILD,
        leftPanelX + 150, leftY, 35, 20, hwnd, (HMENU)ID_AREA_HEIGHT_DEC_100, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-10", WS_VISIBLE | WS_CHILD,
        leftPanelX + 190, leftY, 30, 20, hwnd, (HMENU)ID_AREA_HEIGHT_DEC_10, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"-1", WS_VISIBLE | WS_CHILD,
        leftPanelX + 225, leftY, 25, 20, hwnd, (HMENU)ID_AREA_HEIGHT_DEC_1, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+1", WS_VISIBLE | WS_CHILD,
        leftPanelX + 255, leftY, 25, 20, hwnd, (HMENU)ID_AREA_HEIGHT_INC_1, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+10", WS_VISIBLE | WS_CHILD,
        leftPanelX + 285, leftY, 30, 20, hwnd, (HMENU)ID_AREA_HEIGHT_INC_10, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"+100", WS_VISIBLE | WS_CHILD,
        leftPanelX + 320, leftY, 35, 20, hwnd, (HMENU)ID_AREA_HEIGHT_INC_100, nullptr, nullptr);
    leftY += 35;


    CreateWindow(L"STATIC", L"Rotation:", WS_VISIBLE | WS_CHILD,
        leftPanelX, leftY, 60, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"STATIC", GetRotationText().c_str(), WS_VISIBLE | WS_CHILD | SS_CENTER,
        leftPanelX + 65, leftY, 60, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"Change", WS_VISIBLE | WS_CHILD,
        leftPanelX + 130, leftY, 60, 22, hwnd, (HMENU)ID_ROTATION_TOGGLE, nullptr, nullptr);
    leftY += 35;


    if (monitors.size() > 1) {
        CreateWindow(L"STATIC", L"Monitor:", WS_VISIBLE | WS_CHILD,
            leftPanelX, leftY, 60, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"STATIC", (std::to_wstring(config.currentMonitor + 1) + L"/" + std::to_wstring(monitors.size())).c_str(),
            WS_VISIBLE | WS_CHILD | SS_CENTER, leftPanelX + 65, leftY, 60, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"Switch", WS_VISIBLE | WS_CHILD,
            leftPanelX + 130, leftY, 60, 22, hwnd, (HMENU)ID_MONITOR_SWITCH, nullptr, nullptr);
    }


    int rightY = startButtonY + 20;


    CreateWindow(L"STATIC", L"Prediction:", WS_VISIBLE | WS_CHILD,
        rightPanelX, rightY, 70, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"STATIC", config.movementPrediction ? L"ON" : L"OFF",
        WS_VISIBLE | WS_CHILD | SS_CENTER,
        rightPanelX + 75, rightY, 40, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"Toggle", WS_VISIBLE | WS_CHILD,
        rightPanelX + 120, rightY, 60, 22, hwnd, (HMENU)ID_PREDICTION_TOGGLE, nullptr, nullptr);
    rightY += 30;

    if (config.movementPrediction) {

        CreateWindow(L"STATIC", L"Strength:", WS_VISIBLE | WS_CHILD,
            rightPanelX + 10, rightY, 60, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"STATIC", std::to_wstring(config.predictionStrength / 4).c_str(),
            WS_VISIBLE | WS_CHILD | SS_CENTER,
            rightPanelX + 75, rightY, 30, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"-10", WS_VISIBLE | WS_CHILD,
            rightPanelX + 110, rightY, 30, 20, hwnd, (HMENU)ID_PREDICTION_STR_DEC_10, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"-1", WS_VISIBLE | WS_CHILD,
            rightPanelX + 145, rightY, 25, 20, hwnd, (HMENU)ID_PREDICTION_STR_DEC_1, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+1", WS_VISIBLE | WS_CHILD,
            rightPanelX + 175, rightY, 25, 20, hwnd, (HMENU)ID_PREDICTION_STR_INC_1, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+10", WS_VISIBLE | WS_CHILD,
            rightPanelX + 205, rightY, 30, 20, hwnd, (HMENU)ID_PREDICTION_STR_INC_10, nullptr, nullptr);
        rightY += 25;


        CreateWindow(L"STATIC", L"Max Dist:", WS_VISIBLE | WS_CHILD,
            rightPanelX + 10, rightY, 60, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"STATIC", std::to_wstring(config.maxPredictionDistance).c_str(),
            WS_VISIBLE | WS_CHILD | SS_CENTER,
            rightPanelX + 75, rightY, 40, 20, hwnd, nullptr, nullptr, nullptr);
        rightY += 25;
        CreateWindow(L"BUTTON", L"-100", WS_VISIBLE | WS_CHILD,
            rightPanelX + 10, rightY, 35, 20, hwnd, (HMENU)ID_PREDICTION_DIST_DEC_100, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"-10", WS_VISIBLE | WS_CHILD,
            rightPanelX + 50, rightY, 30, 20, hwnd, (HMENU)ID_PREDICTION_DIST_DEC_10, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"-1", WS_VISIBLE | WS_CHILD,
            rightPanelX + 85, rightY, 25, 20, hwnd, (HMENU)ID_PREDICTION_DIST_DEC_1, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+1", WS_VISIBLE | WS_CHILD,
            rightPanelX + 115, rightY, 25, 20, hwnd, (HMENU)ID_PREDICTION_DIST_INC_1, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+10", WS_VISIBLE | WS_CHILD,
            rightPanelX + 145, rightY, 30, 20, hwnd, (HMENU)ID_PREDICTION_DIST_INC_10, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+100", WS_VISIBLE | WS_CHILD,
            rightPanelX + 180, rightY, 35, 20, hwnd, (HMENU)ID_PREDICTION_DIST_INC_100, nullptr, nullptr);
        rightY += 30;
    }


    CreateWindow(L"STATIC", L"Smoothing:", WS_VISIBLE | WS_CHILD,
        rightPanelX, rightY, 70, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"STATIC", config.smoothingEnabled ? L"ON" : L"OFF",
        WS_VISIBLE | WS_CHILD | SS_CENTER,
        rightPanelX + 75, rightY, 40, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"Toggle", WS_VISIBLE | WS_CHILD,
        rightPanelX + 120, rightY, 60, 22, hwnd, (HMENU)ID_SMOOTHING_TOGGLE, nullptr, nullptr);
    rightY += 30;

    if (config.smoothingEnabled) {
        CreateWindow(L"STATIC", L"Strength:", WS_VISIBLE | WS_CHILD,
            rightPanelX + 10, rightY, 60, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"STATIC", std::to_wstring(config.smoothingStrength / 3).c_str(),
            WS_VISIBLE | WS_CHILD | SS_CENTER,
            rightPanelX + 75, rightY, 30, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"-10", WS_VISIBLE | WS_CHILD,
            rightPanelX + 110, rightY, 30, 20, hwnd, (HMENU)ID_SMOOTHING_STR_DEC_10, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"-1", WS_VISIBLE | WS_CHILD,
            rightPanelX + 145, rightY, 25, 20, hwnd, (HMENU)ID_SMOOTHING_STR_DEC_1, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+1", WS_VISIBLE | WS_CHILD,
            rightPanelX + 175, rightY, 25, 20, hwnd, (HMENU)ID_SMOOTHING_STR_INC_1, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+10", WS_VISIBLE | WS_CHILD,
            rightPanelX + 205, rightY, 30, 20, hwnd, (HMENU)ID_SMOOTHING_STR_INC_10, nullptr, nullptr);
        rightY += 30;
    }


    CreateWindow(L"STATIC", L"Clicks:", WS_VISIBLE | WS_CHILD,
        rightPanelX, rightY, 70, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"STATIC", config.clickEnabled ? L"ON" : L"OFF",
        WS_VISIBLE | WS_CHILD | SS_CENTER,
        rightPanelX + 75, rightY, 40, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"Toggle", WS_VISIBLE | WS_CHILD,
        rightPanelX + 120, rightY, 60, 22, hwnd, (HMENU)ID_CLICK_TOGGLE, nullptr, nullptr);
    rightY += 30;


    CreateWindow(L"STATIC", L"Jitter Reducer:", WS_VISIBLE | WS_CHILD,
        rightPanelX, rightY, 90, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"STATIC", config.jitterReducerEnabled ? L"ON" : L"OFF",
        WS_VISIBLE | WS_CHILD | SS_CENTER,
        rightPanelX + 95, rightY, 40, 20, hwnd, nullptr, nullptr, nullptr);
    CreateWindow(L"BUTTON", L"Toggle", WS_VISIBLE | WS_CHILD,
        rightPanelX + 140, rightY, 60, 22, hwnd, (HMENU)ID_JITTER_REDUCER_TOGGLE, nullptr, nullptr);
    rightY += 30;

    if (config.jitterReducerEnabled) {
        CreateWindow(L"STATIC", L"Threshold:", WS_VISIBLE | WS_CHILD,
            rightPanelX + 10, rightY, 70, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"STATIC", std::to_wstring(config.jitterReducerThreshold).c_str(),
            WS_VISIBLE | WS_CHILD | SS_CENTER,
            rightPanelX + 85, rightY, 30, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"-10", WS_VISIBLE | WS_CHILD,
            rightPanelX + 120, rightY, 30, 20, hwnd, (HMENU)ID_JITTER_REDUCER_DEC_10, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"-1", WS_VISIBLE | WS_CHILD,
            rightPanelX + 155, rightY, 25, 20, hwnd, (HMENU)ID_JITTER_REDUCER_DEC_1, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+1", WS_VISIBLE | WS_CHILD,
            rightPanelX + 185, rightY, 25, 20, hwnd, (HMENU)ID_JITTER_REDUCER_INC_1, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+10", WS_VISIBLE | WS_CHILD,
            rightPanelX + 215, rightY, 30, 20, hwnd, (HMENU)ID_JITTER_REDUCER_INC_10, nullptr, nullptr);
        rightY += 30;
    }
}

std::wstring HighPerformanceTabletDriver::GetRotationText() {
    switch (config.rotation) {
    case 0: return L"0°";
    case 1: return L"90°";
    case 2: return L"180°";
    case 3: return L"270°";
    default: return L"0°";
    }
}

void HighPerformanceTabletDriver::DrawTabletArea(HWND hwnd) {
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hwnd, &ps);

    RECT rect;
    GetClientRect(hwnd, &rect);
    int width = rect.right - rect.left;
    int height = rect.bottom - rect.top;

    HDC memDC = CreateCompatibleDC(hdc);
    HBITMAP memBitmap = CreateCompatibleBitmap(hdc, width, height);
    HBITMAP oldBitmap = (HBITMAP)SelectObject(memDC, memBitmap);

    Graphics graphics(memDC);
    graphics.SetSmoothingMode(SmoothingModeAntiAlias);

    SolidBrush bgBrush(Color(240, 240, 240));
    graphics.FillRectangle(&bgBrush, 0, 0, width, height);

    SolidBrush tabletBrush(Color(100, 100, 100));
    Pen tabletPen(Color(60, 60, 60), 2);
    int tabletMargin = 10;
    graphics.FillRectangle(&tabletBrush, tabletMargin, tabletMargin,
        width - 2 * tabletMargin, height - 2 * tabletMargin);
    graphics.DrawRectangle(&tabletPen, tabletMargin, tabletMargin,
        width - 2 * tabletMargin, height - 2 * tabletMargin);


    float areaWidthRatio = (float)config.areaWidth / currentTablet.widthMM;
    float areaHeightRatio = (float)config.areaHeight / currentTablet.heightMM;

    float areaCenterXRatio = (float)config.areaCenterX / currentTablet.widthMM;
    float areaCenterYRatio = (float)config.areaCenterY / currentTablet.heightMM;

    int availableWidth = width - 2 * tabletMargin - 4;
    int availableHeight = height - 2 * tabletMargin - 4;

    int areaWidth = (int)(availableWidth * areaWidthRatio);
    int areaHeight = (int)(availableHeight * areaHeightRatio);
    int areaCenterX = tabletMargin + 2 + (int)(availableWidth * areaCenterXRatio);
    int areaCenterY = tabletMargin + 2 + (int)(availableHeight * areaCenterYRatio);

    int areaLeft = areaCenterX - areaWidth / 2;
    int areaTop = areaCenterY - areaHeight / 2;

    Color areaColor = running ? Color(100, 255, 100, 120) : Color(255, 255, 100, 120);
    SolidBrush areaBrush(areaColor);
    Pen areaPen(running ? Color(0, 200, 0) : Color(200, 200, 0), 2);

    graphics.FillRectangle(&areaBrush, areaLeft, areaTop, areaWidth, areaHeight);
    graphics.DrawRectangle(&areaPen, areaLeft, areaTop, areaWidth, areaHeight);

    SolidBrush centerBrush(Color(255, 0, 0));
    graphics.FillEllipse(&centerBrush, areaCenterX - 3, areaCenterY - 3, 6, 6);

    std::wstring sizeText;
    sizeText = std::to_wstring(config.areaWidth) + L"×" + std::to_wstring(config.areaHeight) + L"mm";

    FontFamily fontFamily(L"Segoe UI");
    Font font(&fontFamily, 9, FontStyleRegular, UnitPoint);
    SolidBrush textBrush(Color(0, 0, 0));
    graphics.DrawString(sizeText.c_str(), -1, &font, PointF(5, 5), &textBrush);

    BitBlt(hdc, 0, 0, width, height, memDC, 0, 0, SRCCOPY);

    SelectObject(memDC, oldBitmap);
    DeleteObject(memBitmap);
    DeleteDC(memDC);

    EndPaint(hwnd, &ps);
}

void HighPerformanceTabletDriver::HandleCommand(int commandId) {
    switch (commandId) {

    case ID_CENTER_UP: MoveAreaCenter(0, -1); break;
    case ID_CENTER_DOWN: MoveAreaCenter(0, 1); break;
    case ID_CENTER_LEFT: MoveAreaCenter(-1, 0); break;
    case ID_CENTER_RIGHT: MoveAreaCenter(1, 0); break;

    case ID_CENTER_X_DEC_1: MoveAreaCenter(-1, 0); break;
    case ID_CENTER_X_INC_1: MoveAreaCenter(1, 0); break;
    case ID_CENTER_X_DEC_10: MoveAreaCenter(-10, 0); break;
    case ID_CENTER_X_INC_10: MoveAreaCenter(10, 0); break;
    case ID_CENTER_X_DEC_100: MoveAreaCenter(-100, 0); break;
    case ID_CENTER_X_INC_100: MoveAreaCenter(100, 0); break;


    case ID_CENTER_Y_DEC_1: MoveAreaCenter(0, -1); break;
    case ID_CENTER_Y_INC_1: MoveAreaCenter(0, 1); break;
    case ID_CENTER_Y_DEC_10: MoveAreaCenter(0, -10); break;
    case ID_CENTER_Y_INC_10: MoveAreaCenter(0, 10); break;
    case ID_CENTER_Y_DEC_100: MoveAreaCenter(0, -100); break;
    case ID_CENTER_Y_INC_100: MoveAreaCenter(0, 100); break;


    case ID_AREA_WIDTH_DEC_1: AdjustAreaSize(-1, 0); break;
    case ID_AREA_WIDTH_INC_1: AdjustAreaSize(1, 0); break;
    case ID_AREA_WIDTH_DEC_10: AdjustAreaSize(-10, 0); break;
    case ID_AREA_WIDTH_INC_10: AdjustAreaSize(10, 0); break;
    case ID_AREA_WIDTH_DEC_50: AdjustAreaSize(-50, 0); break;
    case ID_AREA_WIDTH_INC_50: AdjustAreaSize(50, 0); break;
    case ID_AREA_WIDTH_DEC_100: AdjustAreaSize(-100, 0); break;
    case ID_AREA_WIDTH_INC_100: AdjustAreaSize(100, 0); break;


    case ID_AREA_HEIGHT_DEC_1: AdjustAreaSize(0, -1); break;
    case ID_AREA_HEIGHT_INC_1: AdjustAreaSize(0, 1); break;
    case ID_AREA_HEIGHT_DEC_10: AdjustAreaSize(0, -10); break;
    case ID_AREA_HEIGHT_INC_10: AdjustAreaSize(0, 10); break;
    case ID_AREA_HEIGHT_DEC_50: AdjustAreaSize(0, -50); break;
    case ID_AREA_HEIGHT_INC_50: AdjustAreaSize(0, 50); break;
    case ID_AREA_HEIGHT_DEC_100: AdjustAreaSize(0, -100); break;
    case ID_AREA_HEIGHT_INC_100: AdjustAreaSize(0, 100); break;

    case ID_ROTATION_TOGGLE: ConfigureRotation(); break;
    case ID_MONITOR_SWITCH: SwitchMonitor(); break;
    case ID_START_STOP:
        if (running) Stop();
        else Start();
        break;


    case ID_PREDICTION_TOGGLE: TogglePrediction(); break;
    case ID_PREDICTION_STR_DEC_1: AdjustPredictionStrength(-4); break;
    case ID_PREDICTION_STR_INC_1: AdjustPredictionStrength(4); break;
    case ID_PREDICTION_STR_DEC_10: AdjustPredictionStrength(-40); break;
    case ID_PREDICTION_STR_INC_10: AdjustPredictionStrength(40); break;

    case ID_PREDICTION_DIST_DEC_1: AdjustPredictionDistance(-1); break;
    case ID_PREDICTION_DIST_INC_1: AdjustPredictionDistance(1); break;
    case ID_PREDICTION_DIST_DEC_10: AdjustPredictionDistance(-10); break;
    case ID_PREDICTION_DIST_INC_10: AdjustPredictionDistance(10); break;
    case ID_PREDICTION_DIST_DEC_100: AdjustPredictionDistance(-100); break;
    case ID_PREDICTION_DIST_INC_100: AdjustPredictionDistance(100); break;

    case ID_CLICK_TOGGLE: ToggleClick(); break;


    case ID_SMOOTHING_TOGGLE: ToggleSmoothing(); break;
    case ID_SMOOTHING_STR_DEC_1: AdjustSmoothingStrength(-3); break;
    case ID_SMOOTHING_STR_INC_1: AdjustSmoothingStrength(3); break;
    case ID_SMOOTHING_STR_DEC_10: AdjustSmoothingStrength(-30); break;
    case ID_SMOOTHING_STR_INC_10: AdjustSmoothingStrength(30); break;


    case ID_JITTER_REDUCER_TOGGLE: ToggleJitterReducer(); break;
    case ID_JITTER_REDUCER_DEC_1: AdjustJitterReducer(-1); break;
    case ID_JITTER_REDUCER_INC_1: AdjustJitterReducer(1); break;
    case ID_JITTER_REDUCER_DEC_10: AdjustJitterReducer(-10); break;
    case ID_JITTER_REDUCER_INC_10: AdjustJitterReducer(10); break;
    }

    UpdateGUI();
}

void HighPerformanceTabletDriver::UpdateGUI() {
    if (!mainWindow) return;


    EnumChildWindows(mainWindow, [](HWND hwnd, LPARAM) -> BOOL {
        DestroyWindow(hwnd);
        return TRUE;
        }, 0);


    CreateControls(mainWindow);


    InvalidateRect(mainWindow, nullptr, TRUE);
    UpdateWindow(mainWindow);
}


void HighPerformanceTabletDriver::LoadConfig() {
    std::ifstream file("driver.config");
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key, equals, value;
            if (iss >> key >> equals >> value && equals == "=") {
                if (key == "areaWidth") config.areaWidth = std::stoi(value);
                else if (key == "areaHeight") config.areaHeight = std::stoi(value);
                else if (key == "areaCenterX") config.areaCenterX = std::stoi(value);
                else if (key == "areaCenterY") config.areaCenterY = std::stoi(value);
                else if (key == "rotation") config.rotation = std::stoi(value);
                else if (key == "movementPrediction") config.movementPrediction = (value == "1");
                else if (key == "predictionStrength") config.predictionStrength = std::stoi(value);
                else if (key == "maxPredictionDistance") config.maxPredictionDistance = std::stoi(value);
                else if (key == "clickEnabled") config.clickEnabled = (value == "1");
                else if (key == "currentMonitor") config.currentMonitor = std::stoi(value);
                else if (key == "smoothingEnabled") config.smoothingEnabled = (value == "1");
                else if (key == "smoothingStrength") config.smoothingStrength = std::stoi(value);
                else if (key == "jitterReducerEnabled") config.jitterReducerEnabled = (value == "1");
                else if (key == "jitterReducerThreshold") config.jitterReducerThreshold = std::stoi(value);
            }
        }
        file.close();
    }
}

void HighPerformanceTabletDriver::SaveConfig() {
    std::ofstream file("driver.config");
    if (file.is_open()) {
        file << "areaWidth = " << config.areaWidth << std::endl;
        file << "areaHeight = " << config.areaHeight << std::endl;
        file << "areaCenterX = " << config.areaCenterX << std::endl;
        file << "areaCenterY = " << config.areaCenterY << std::endl;
        file << "rotation = " << config.rotation << std::endl;
        file << "movementPrediction = " << (config.movementPrediction ? 1 : 0) << std::endl;
        file << "predictionStrength = " << config.predictionStrength << std::endl;
        file << "maxPredictionDistance = " << config.maxPredictionDistance << std::endl;
        file << "clickEnabled = " << (config.clickEnabled ? 1 : 0) << std::endl;
        file << "currentMonitor = " << config.currentMonitor << std::endl;
        file << "smoothingEnabled = " << (config.smoothingEnabled ? 1 : 0) << std::endl;
        file << "smoothingStrength = " << config.smoothingStrength << std::endl;
        file << "jitterReducerEnabled = " << (config.jitterReducerEnabled ? 1 : 0) << std::endl;
        file << "jitterReducerThreshold = " << config.jitterReducerThreshold << std::endl;
        file.close();
    }
}

bool HighPerformanceTabletDriver::Initialize() {
    GUID hidGuid;
    HidD_GetHidGuid(&hidGuid);

    HDEVINFO deviceInfoSet = SetupDiGetClassDevs(&hidGuid, NULL, NULL,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);

    if (deviceInfoSet == INVALID_HANDLE_VALUE) {
        return false;
    }

    SP_DEVICE_INTERFACE_DATA deviceInterfaceData;
    deviceInterfaceData.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);

    for (DWORD deviceIndex = 0;
        SetupDiEnumDeviceInterfaces(deviceInfoSet, NULL, &hidGuid,
            deviceIndex, &deviceInterfaceData);
        deviceIndex++) {

        DWORD requiredSize = 0;
        SetupDiGetDeviceInterfaceDetail(deviceInfoSet, &deviceInterfaceData,
            NULL, 0, &requiredSize, NULL);

        PSP_DEVICE_INTERFACE_DETAIL_DATA deviceInterfaceDetailData =
            (PSP_DEVICE_INTERFACE_DETAIL_DATA)malloc(requiredSize);

        if (deviceInterfaceDetailData == nullptr) {
            continue;
        }

        deviceInterfaceDetailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

        if (SetupDiGetDeviceInterfaceDetail(deviceInfoSet, &deviceInterfaceData,
            deviceInterfaceDetailData, requiredSize,
            NULL, NULL)) {

            HANDLE testHandle = CreateFile(deviceInterfaceDetailData->DevicePath,
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                NULL, OPEN_EXISTING, 0, NULL);

            if (testHandle != INVALID_HANDLE_VALUE) {
                HIDD_ATTRIBUTES attributes;
                attributes.Size = sizeof(HIDD_ATTRIBUTES);

                if (HidD_GetAttributes(testHandle, &attributes)) {
                    TabletType detectedType = DetectTabletType(attributes.VendorID, attributes.ProductID);

                    if (detectedType != TabletType::UNKNOWN) {
                        std::cout << "Found tablet: " << TABLET_SPECS[(int)detectedType].name
                            << " (VID: 0x" << std::hex << attributes.VendorID
                            << " PID: 0x" << attributes.ProductID << std::dec << ")" << std::endl;

                        currentTablet = TABLET_SPECS[(int)detectedType];
                        deviceHandle = testHandle;

                        if (config.areaWidth == 28 && config.areaHeight == 22) {
                            SetDefaultArea();
                        }

                        OptimizeSystemForTablet();
                        InitializeTabletParser();
                        UpdatePreCalculatedConstants();

                        free(deviceInterfaceDetailData);
                        SetupDiDestroyDeviceInfoList(deviceInfoSet);
                        return true;
                    }
                }
                CloseHandle(testHandle);
            }
        }
        free(deviceInterfaceDetailData);
    }

    SetupDiDestroyDeviceInfoList(deviceInfoSet);
    return false;
}

void HighPerformanceTabletDriver::MoveAreaCenter(int deltaX, int deltaY) {
    config.areaCenterX += deltaX;
    config.areaCenterY += deltaY;


    config.areaCenterX = max(config.areaWidth / 2, min(currentTablet.widthMM - config.areaWidth / 2, config.areaCenterX));
    config.areaCenterY = max(config.areaHeight / 2, min(currentTablet.heightMM - config.areaHeight / 2, config.areaCenterY));

    SaveConfig();
    UpdatePreCalculatedConstants();
}

void HighPerformanceTabletDriver::AdjustAreaSize(int widthDelta, int heightDelta) {

    config.areaWidth = max(5, min(currentTablet.widthMM, config.areaWidth + widthDelta));
    config.areaHeight = max(5, min(currentTablet.heightMM, config.areaHeight + heightDelta));


    config.areaCenterX = max(config.areaWidth / 2, min(currentTablet.widthMM - config.areaWidth / 2, config.areaCenterX));
    config.areaCenterY = max(config.areaHeight / 2, min(currentTablet.heightMM - config.areaHeight / 2, config.areaCenterY));

    SaveConfig();
    UpdatePreCalculatedConstants();
}

void HighPerformanceTabletDriver::ConfigureRotation() {
    config.rotation = (config.rotation + 1) % 4;
    SaveConfig();
    UpdatePreCalculatedConstants();


    CheckAndCorrectAreaBounds();
}

void HighPerformanceTabletDriver::CheckAndCorrectAreaBounds() {

    int minX = config.areaWidth / 2;
    int maxX = currentTablet.widthMM - config.areaWidth / 2;
    int minY = config.areaHeight / 2;
    int maxY = currentTablet.heightMM - config.areaHeight / 2;


    bool corrected = false;
    if (config.areaCenterX < minX) {
        config.areaCenterX = minX;
        corrected = true;
    }
    else if (config.areaCenterX > maxX) {
        config.areaCenterX = maxX;
        corrected = true;
    }

    if (config.areaCenterY < minY) {
        config.areaCenterY = minY;
        corrected = true;
    }
    else if (config.areaCenterY > maxY) {
        config.areaCenterY = maxY;
        corrected = true;
    }


    if (corrected) {
        SaveConfig();
        UpdatePreCalculatedConstants();
    }
}

void HighPerformanceTabletDriver::SwitchMonitor() {
    if (monitors.size() > 1) {
        config.currentMonitor = (config.currentMonitor + 1) % monitors.size();
        SaveConfig();
    }
}

void HighPerformanceTabletDriver::TogglePrediction() {
    config.movementPrediction = !config.movementPrediction;
    SaveConfig();
    UpdatePreCalculatedConstants();
}

void HighPerformanceTabletDriver::ToggleSmoothing() {
    config.smoothingEnabled = !config.smoothingEnabled;
    if (!config.smoothingEnabled) {
        hotData.hasSmoothedData.store(false);
    }
    SaveConfig();
    UpdatePreCalculatedConstants();
}

void HighPerformanceTabletDriver::AdjustSmoothingStrength(int delta) {
    config.smoothingStrength = max(3, min(300, config.smoothingStrength + delta));
    SaveConfig();
    UpdatePreCalculatedConstants();
}

void HighPerformanceTabletDriver::ToggleClick() {
    config.clickEnabled = !config.clickEnabled;
    SaveConfig();
}

void HighPerformanceTabletDriver::AdjustPredictionStrength(int delta) {
    config.predictionStrength = max(4, min(400, config.predictionStrength + delta));
    SaveConfig();
    UpdatePreCalculatedConstants();
}

void HighPerformanceTabletDriver::ToggleJitterReducer() {
    config.jitterReducerEnabled = !config.jitterReducerEnabled;
    SaveConfig();
}

void HighPerformanceTabletDriver::AdjustJitterReducer(int delta) {
    config.jitterReducerThreshold = max(1, min(50, config.jitterReducerThreshold + delta));
    SaveConfig();
}

void HighPerformanceTabletDriver::AdjustPredictionDistance(int delta) {
    config.maxPredictionDistance = max(10, min(1000, config.maxPredictionDistance + delta));
    SaveConfig();
}


void HighPerformanceTabletDriver::RestartDriver() {
    if (running) {
        Stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    Start();
}

void HighPerformanceTabletDriver::RunConfiguration() {
    HWND consoleWindow = GetConsoleWindow();
    if (consoleWindow) {
        ShowWindow(consoleWindow, SW_HIDE);
    }

    configMode = true;
    guiRunning = true;

    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);

    SetOptimalThreadAffinity(2);

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = L"UltraTabletDriverGUI";
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    RegisterClass(&wc);

    mainWindow = CreateWindow(
        L"UltraTabletDriverGUI",
        L"Ultra Tablet Driver v4.2 - Optimized Edition",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, 1000, 500,
        nullptr, nullptr, GetModuleHandle(nullptr), this
    );

    if (!mainWindow) return;

    ShowWindow(mainWindow, SW_SHOW);
    UpdateWindow(mainWindow);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0) && guiRunning) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    configMode = false;
    guiRunning = false;

    if (guiThread.joinable()) {
        guiThread.join();
    }
}

void HighPerformanceTabletDriver::WaitForExit() {
    if (configThread.joinable()) {
        configThread.join();
    }
}

int main() {
    std::cout << "Ultra Tablet Driver v4.3 - Hyper-Optimized Edition" << std::endl;
    std::cout << "Optimizations: Memory layout, SIMD vectorization, branch prediction, threading, algorithms" << std::endl;
    std::cout << "Support for Wacom CTL-672, CTL-472 & XPPen Star G640" << std::endl << std::endl;

    if (!SIMDMath::HasSSE2()) {
        std::cout << "Warning: SSE2 not supported on this CPU. Performance will be reduced." << std::endl;
    }

    timeBeginPeriod(1);

    HighPerformanceTabletDriver driver;

    if (!driver.Initialize()) {
        std::cerr << "Failed to initialize tablet driver" << std::endl;
        std::cout << "Make sure your supported tablet is connected:" << std::endl;
        std::cout << "- Wacom CTL-672 (One by Wacom Medium)" << std::endl;
        std::cout << "- Wacom CTL-472 (One by Wacom Small)" << std::endl;
        std::cout << "- XPPen Star G 640" << std::endl;
        std::cout << "If your tablet is connected and all drivers are disabled:" << std::endl;
        std::cout << "- Install and run OpenTabletDriver and then close it (make sure to close it in system tray too)" << std::endl;
        std::cout << "- Make sure you are running this app as Administrator" << std::endl;
        std::cout << "If nothing helps, contact me on Discord: vilounos" << std::endl;
        system("pause");
        timeEndPeriod(1);
        return 1;
    }

    try {
        driver.RunConfiguration();
        driver.WaitForExit();
    }
    catch (...) {
        std::cout << "Emergency shutdown due to unexpected error" << std::endl;
    }

    std::cout << "Stopping Ultra Tablet Driver..." << std::endl;
    driver.Stop();

    timeEndPeriod(1);

    return 0;
}