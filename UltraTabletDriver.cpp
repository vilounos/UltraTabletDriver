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


struct alignas(16) Vec2f {
    float x, y;
    float _pad[2];

    Vec2f() : x(0.0f), y(0.0f), _pad{ 0.0f, 0.0f } {}
    Vec2f(float x_, float y_) : x(x_), y(y_), _pad{ 0.0f, 0.0f } {}

    inline void Prefetch() const {
        _mm_prefetch(reinterpret_cast<const char*>(this), _MM_HINT_T0);
    }
};

struct alignas(16) Vec4f {
    float x, y, z, w;

    Vec4f() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    Vec4f(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

struct alignas(16) Vec2i {
    int x, y;
    int _pad[2];

    Vec2i() : x(0), y(0), _pad{ 0, 0 } {}
    Vec2i(int x_, int y_) : x(x_), y(y_), _pad{ 0, 0 } {}

    inline void Prefetch() const {
        _mm_prefetch(reinterpret_cast<const char*>(this), _MM_HINT_T0);
    }
};


class SIMDMath {
public:
    static bool HasSSE() {
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        return (cpuInfo[3] & (1 << 25)) != 0;
    }

    static bool HasSSE2() {
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        return (cpuInfo[3] & (1 << 26)) != 0;
    }

    static bool HasAVX() {
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        return (cpuInfo[2] & (1 << 28)) != 0;
    }

    static inline Vec2f Add(const Vec2f& a, const Vec2f& b) {
        __m128 va = _mm_load_ps(reinterpret_cast<const float*>(&a));
        __m128 vb = _mm_load_ps(reinterpret_cast<const float*>(&b));
        __m128 result = _mm_add_ps(va, vb);

        Vec2f output;
        _mm_store_ps(reinterpret_cast<float*>(&output), result);
        return output;
    }

    static inline Vec2f Sub(const Vec2f& a, const Vec2f& b) {
        __m128 va = _mm_load_ps(reinterpret_cast<const float*>(&a));
        __m128 vb = _mm_load_ps(reinterpret_cast<const float*>(&b));
        __m128 result = _mm_sub_ps(va, vb);

        Vec2f output;
        _mm_store_ps(reinterpret_cast<float*>(&output), result);
        return output;
    }

    static inline Vec2f Mul(const Vec2f& a, const Vec2f& b) {
        __m128 va = _mm_load_ps(reinterpret_cast<const float*>(&a));
        __m128 vb = _mm_load_ps(reinterpret_cast<const float*>(&b));
        __m128 result = _mm_mul_ps(va, vb);

        Vec2f output;
        _mm_store_ps(reinterpret_cast<float*>(&output), result);
        return output;
    }

    static inline Vec2f Scale(const Vec2f& a, float scale) {
        __m128 va = _mm_load_ps(reinterpret_cast<const float*>(&a));
        __m128 vscale = _mm_set1_ps(scale);
        __m128 result = _mm_mul_ps(va, vscale);

        Vec2f output;
        _mm_store_ps(reinterpret_cast<float*>(&output), result);
        return output;
    }

    static inline float Dot(const Vec2f& a, const Vec2f& b) {
        __m128 va = _mm_load_ps(reinterpret_cast<const float*>(&a));
        __m128 vb = _mm_load_ps(reinterpret_cast<const float*>(&b));
        __m128 mul = _mm_mul_ps(va, vb);
        __m128 hadd = _mm_hadd_ps(mul, mul);
        return _mm_cvtss_f32(hadd);
    }

    static inline float Length(const Vec2f& a) {
        __m128 va = _mm_load_ps(reinterpret_cast<const float*>(&a));
        __m128 mul = _mm_mul_ps(va, va);
        __m128 hadd = _mm_hadd_ps(mul, mul);
        __m128 sqrt_result = _mm_sqrt_ss(hadd);
        return _mm_cvtss_f32(sqrt_result);
    }

    static inline Vec2f Lerp(const Vec2f& a, const Vec2f& b, float t) {
        __m128 va = _mm_load_ps(reinterpret_cast<const float*>(&a));
        __m128 vb = _mm_load_ps(reinterpret_cast<const float*>(&b));
        __m128 vt = _mm_set1_ps(t);
        __m128 diff = _mm_sub_ps(vb, va);
        __m128 lerp = _mm_mul_ps(diff, vt);
        __m128 result = _mm_add_ps(va, lerp);

        Vec2f output;
        _mm_store_ps(reinterpret_cast<float*>(&output), result);
        return output;
    }

    static inline Vec2f ApplyRotation(const Vec2f& point, float cosAngle, float sinAngle) {
        Vec2f result;
        result.x = point.x * cosAngle - point.y * sinAngle;
        result.y = point.x * sinAngle + point.y * cosAngle;
        return result;
    }

    static inline Vec2f NormalizeCoords(const Vec2i& raw, const Vec2i& areaMin, const Vec2i& areaSize) {
        __m128i vraw = _mm_load_si128(reinterpret_cast<const __m128i*>(&raw));
        __m128i vmin = _mm_load_si128(reinterpret_cast<const __m128i*>(&areaMin));
        __m128i vsize = _mm_load_si128(reinterpret_cast<const __m128i*>(&areaSize));

        __m128i diff = _mm_sub_epi32(vraw, vmin);

        __m128 fdiff = _mm_cvtepi32_ps(diff);
        __m128 fsize = _mm_cvtepi32_ps(vsize);

        __m128 normalized = _mm_div_ps(fdiff, fsize);

        Vec2f result;
        _mm_store_ps(reinterpret_cast<float*>(&result), normalized);
        return result;
    }

    static inline Vec2i MapToScreen(const Vec2f& normalized, const Vec2i& screenSize, const Vec2i& screenOffset) {
        Vec2i result;
        result.x = (int)(normalized.x * screenSize.x) + screenOffset.x;
        result.y = (int)(normalized.y * screenSize.y) + screenOffset.y;
        return result;
    }
};

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
    alignas(64) volatile uint64_t writeIndex;
    alignas(64) volatile uint64_t readIndex;

    static constexpr uint64_t MASK = Size - 1;

public:
    LockFreeRingBuffer() : writeIndex(0), readIndex(0) {}


    __forceinline bool TryPush(const T& item) {
        const uint64_t currentWrite = writeIndex;
        const uint64_t nextWrite = currentWrite + 1;

        if ((nextWrite & MASK) == (readIndex & MASK)) {
            return false;
        }

        buffer[currentWrite & MASK] = item;


        _mm_sfence();
        writeIndex = nextWrite;

        return true;
    }


    __forceinline bool TryPop(T& item) {
        const uint64_t currentRead = readIndex;

        if (currentRead == writeIndex) {
            return false;
        }

        item = buffer[currentRead & MASK];


        _mm_lfence();
        readIndex = currentRead + 1;

        return true;
    }

    __forceinline bool IsEmpty() const {
        return readIndex == writeIndex;
    }

    __forceinline size_t Size() const {
        return (writeIndex - readIndex) & MASK;
    }
};

struct DriverConfig {
    int areaWidth = 28;
    int areaHeight = 22;
    int areaCenterX = 80;
    int areaCenterY = 80;
    int rotation = 0;
    bool movementPrediction = false;
    int predictionStrength = 2;
    bool clickEnabled = false;
    int currentMonitor = 0;

    bool smoothingEnabled = false;
    int smoothingStrength = 5;
};


struct alignas(16) TabletData {
    Vec2i rawPos;
    bool inProximity;
    bool isTouching;
    bool isValid;
    float _pad;
    std::chrono::high_resolution_clock::time_point timestamp;
};

template<typename T, size_t PoolSize>
class alignas(64) ObjectPool {
private:
    alignas(64) T pool[PoolSize];
    alignas(64) std::atomic<uint32_t> freeList[PoolSize];
    alignas(64) std::atomic<uint32_t> freeHead{ 0 };
    alignas(64) std::atomic<uint32_t> allocCount{ 0 };

public:
    ObjectPool() {
        for (uint32_t i = 0; i < PoolSize; i++) {
            freeList[i].store(i);
        }
    }

    T* Acquire() {
        uint32_t head = freeHead.load(std::memory_order_relaxed);
        while (head < PoolSize) {
            if (freeHead.compare_exchange_weak(head, head + 1, std::memory_order_acq_rel)) {
                allocCount.fetch_add(1, std::memory_order_relaxed);
                return &pool[freeList[head].load()];
            }
        }
        return nullptr;
    }

    void Release(T* obj) {
        if (!obj) return;
        uint32_t index = static_cast<uint32_t>(obj - pool);
        if (index >= PoolSize) return;

        uint32_t head = freeHead.fetch_sub(1, std::memory_order_acq_rel) - 1;
        freeList[head].store(index, std::memory_order_release);
        allocCount.fetch_sub(1, std::memory_order_relaxed);
    }

    size_t GetAllocatedCount() const {
        return allocCount.load(std::memory_order_relaxed);
    }
};

template<typename T, size_t PoolSize, size_t Alignment = 64>
class alignas(64) HighPerformancePool {
private:

    alignas(Alignment) char storage[PoolSize * sizeof(T)];
    alignas(64) std::atomic<uint32_t> freeStack[PoolSize];
    alignas(64) std::atomic<uint32_t> freeTop{ 0 };
    alignas(64) std::atomic<uint32_t> allocCount{ 0 };


    alignas(64) volatile uint32_t nextHint { 0 };

public:
    HighPerformancePool() {

        for (uint32_t i = 0; i < PoolSize; i++) {
            freeStack[i].store(i, std::memory_order_relaxed);
        }


        for (size_t i = 0; i < min(PoolSize * sizeof(T), 4 * 64); i += 64) {
            _mm_prefetch(storage + i, _MM_HINT_T1);
        }
    }

    __forceinline T* AcquireFast() {
        uint32_t top = freeTop.load(std::memory_order_acquire);

        if (top >= PoolSize) {
            return nullptr;
        }


        if (!freeTop.compare_exchange_weak(top, top + 1,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
            return nullptr;
        }

        uint32_t index = freeStack[top].load(std::memory_order_relaxed);
        T* ptr = reinterpret_cast<T*>(storage + index * sizeof(T));


        if (top + 1 < PoolSize) {
            uint32_t nextIndex = freeStack[top + 1].load(std::memory_order_relaxed);
            _mm_prefetch(storage + nextIndex * sizeof(T), _MM_HINT_T0);
        }

        allocCount.fetch_add(1, std::memory_order_relaxed);
        return ptr;
    }

    __forceinline void ReleaseFast(T* ptr) {
        if (!ptr) return;

        uint32_t index = static_cast<uint32_t>((reinterpret_cast<char*>(ptr) - storage) / sizeof(T));
        if (index >= PoolSize) return;

        uint32_t top = freeTop.fetch_sub(1, std::memory_order_acq_rel) - 1;
        freeStack[top].store(index, std::memory_order_release);
        allocCount.fetch_sub(1, std::memory_order_relaxed);
    }

    size_t GetAllocatedCount() const {
        return allocCount.load(std::memory_order_relaxed);
    }

    bool IsFull() const {
        return freeTop.load(std::memory_order_acquire) >= PoolSize;
    }
};

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

    __forceinline void AddSample(float vx, float vy, uint64_t timestamp) {
        int idx = writeIndex.load(std::memory_order_relaxed);

        velocityX[idx] = vx;
        velocityY[idx] = vy;
        timestamps[idx] = timestamp;
        weights[idx] = 1.0f;

        writeIndex.store((idx + 1) % HISTORY_SIZE, std::memory_order_release);

        int currentCount = count.load(std::memory_order_relaxed);
        if (currentCount < HISTORY_SIZE) {
            count.store(currentCount + 1, std::memory_order_release);
        }
    }

    __forceinline Vec2f CalculateWeightedAverage() const {
        int sampleCount = count.load(std::memory_order_acquire);
        if (sampleCount == 0) return Vec2f(0.0f, 0.0f);


        _mm_prefetch(reinterpret_cast<const char*>(velocityX), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(velocityY), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(weights), _MM_HINT_T0);

        __m128 sumX = _mm_setzero_ps();
        __m128 sumY = _mm_setzero_ps();
        __m128 totalWeight = _mm_setzero_ps();


        int simdCount = (sampleCount / 4) * 4;
        for (int i = 0; i < simdCount; i += 4) {
            __m128 vx = _mm_load_ps(&velocityX[i]);
            __m128 vy = _mm_load_ps(&velocityY[i]);
            __m128 w = _mm_load_ps(&weights[i]);

            sumX = _mm_add_ps(sumX, _mm_mul_ps(vx, w));
            sumY = _mm_add_ps(sumY, _mm_mul_ps(vy, w));
            totalWeight = _mm_add_ps(totalWeight, w);
        }


        for (int i = simdCount; i < sampleCount; i++) {
            sumX = _mm_add_ss(sumX, _mm_mul_ss(_mm_load_ss(&velocityX[i]), _mm_load_ss(&weights[i])));
            sumY = _mm_add_ss(sumY, _mm_mul_ss(_mm_load_ss(&velocityY[i]), _mm_load_ss(&weights[i])));
            totalWeight = _mm_add_ss(totalWeight, _mm_load_ss(&weights[i]));
        }


        sumX = _mm_hadd_ps(sumX, sumX);
        sumX = _mm_hadd_ps(sumX, sumX);
        sumY = _mm_hadd_ps(sumY, sumY);
        sumY = _mm_hadd_ps(sumY, sumY);
        totalWeight = _mm_hadd_ps(totalWeight, totalWeight);
        totalWeight = _mm_hadd_ps(totalWeight, totalWeight);

        float weightSum = _mm_cvtss_f32(totalWeight);
        if (weightSum > 0.0f) {
            return Vec2f(_mm_cvtss_f32(sumX) / weightSum, _mm_cvtss_f32(sumY) / weightSum);
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
    alignas(16) std::atomic<Vec2i> smoothedRawPos;
    std::atomic<bool> hasSmoothedData{ false };
    static constexpr float SMOOTHING_THRESHOLD_BASE = 2.0f;
    static HighPerformancePool<TabletData, 128> tabletPool;


    static constexpr size_t BATCH_BUFFER_SIZE = 8;
    LockFreeRingBuffer<TabletData, BATCH_BUFFER_SIZE> batchBuffer;


    struct alignas(64) TabletLookupTable {
        float scaleX, scaleY;
        int maxX, maxY;
        int centerOffsetX, centerOffsetY;
    } tabletLookup;


    VelocityHistorySOA velocityHistory;


    alignas(16) Vec2f workingPositions[32];
    alignas(16) float workingWeights[32];
    alignas(16) uint64_t workingTimestamps[32];


    DWORD_PTR systemAffinityMask;
    DWORD_PTR processAffinityMask;
    std::vector<int> availableCores;


    std::chrono::high_resolution_clock::time_point lastPositionTime;
    std::chrono::high_resolution_clock::time_point secondLastPositionTime;


private:
    static constexpr size_t TABLET_BUFFER_SIZE = 32;
    LockFreeRingBuffer<TabletData, TABLET_BUFFER_SIZE> tabletDataBuffer;

    alignas(64) TabletData fastAccessData;
    volatile bool fastDataValid = false;

private:

    HWND mainWindow = nullptr;
    HWND statusLabel = nullptr;
    HWND visualArea = nullptr;
    std::thread guiThread;
    std::atomic<bool> guiRunning{ false };
    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;


    enum {

        ID_CENTER_UP = 1001, ID_CENTER_DOWN, ID_CENTER_LEFT, ID_CENTER_RIGHT,


        ID_AREA_WIDTH_DEC, ID_AREA_WIDTH_INC,
        ID_AREA_HEIGHT_DEC, ID_AREA_HEIGHT_INC,


        ID_ROTATION_TOGGLE, ID_MONITOR_SWITCH,
        ID_START_STOP,


        ID_PREDICTION_TOGGLE, ID_PREDICTION_STR_DEC, ID_PREDICTION_STR_INC,
        ID_CLICK_TOGGLE,

        ID_SMOOTHING_TOGGLE, ID_SMOOTHING_STR_DEC, ID_SMOOTHING_STR_INC,

        ID_VISUAL_AREA = 2000
    };

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

    LRESULT HandleWindowMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
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

    void CreateControls(HWND hwnd) {
        int leftCol = 20;
        int rightCol = 280;
        int y = 20;


        CreateWindow(L"STATIC", L"═══ TABLET AREA ═══", WS_VISIBLE | WS_CHILD | SS_CENTER,
            leftCol, y, 240, 25, hwnd, nullptr, nullptr, nullptr);
        y += 35;


        WNDCLASS visualClass = {};
        visualClass.lpfnWndProc = VisualAreaProc;
        visualClass.hInstance = GetModuleHandle(nullptr);
        visualClass.lpszClassName = L"VisualArea";
        visualClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
        RegisterClass(&visualClass);

        visualArea = CreateWindow(L"VisualArea", L"", WS_VISIBLE | WS_CHILD | WS_BORDER,
            leftCol + 20, y, 200, 150, hwnd, (HMENU)ID_VISUAL_AREA, nullptr, nullptr);
        SetWindowLongPtr(visualArea, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
        y += 160;


        CreateWindow(L"BUTTON", L"↑", WS_VISIBLE | WS_CHILD,
            leftCol + 85, y, 30, 25, hwnd, (HMENU)ID_CENTER_UP, nullptr, nullptr);
        y += 30;

        CreateWindow(L"BUTTON", L"←", WS_VISIBLE | WS_CHILD,
            leftCol + 50, y, 30, 25, hwnd, (HMENU)ID_CENTER_LEFT, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"→", WS_VISIBLE | WS_CHILD,
            leftCol + 120, y, 30, 25, hwnd, (HMENU)ID_CENTER_RIGHT, nullptr, nullptr);
        y += 30;

        CreateWindow(L"BUTTON", L"↓", WS_VISIBLE | WS_CHILD,
            leftCol + 85, y, 30, 25, hwnd, (HMENU)ID_CENTER_DOWN, nullptr, nullptr);
        y += 40;


        CreateWindow(L"STATIC", L"Position:", WS_VISIBLE | WS_CHILD,
            leftCol, y, 100, 20, hwnd, nullptr, nullptr, nullptr);
        y += 25;
        std::wstring posText = L"X: " + std::to_wstring(config.areaCenterX) +
            L"mm, Y: " + std::to_wstring(config.areaCenterY) + L"mm";
        CreateWindow(L"STATIC", posText.c_str(), WS_VISIBLE | WS_CHILD | SS_CENTER,
            leftCol, y, 200, 20, hwnd, nullptr, nullptr, nullptr);
        y += 35;


        CreateWindow(L"STATIC", L"Area width:", WS_VISIBLE | WS_CHILD,
            leftCol, y, 100, 20, hwnd, nullptr, nullptr, nullptr);
        y += 25;
        CreateWindow(L"STATIC", (std::to_wstring(config.areaWidth) + L" mm").c_str(),
            WS_VISIBLE | WS_CHILD | SS_CENTER,
            leftCol + 20, y, 60, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"-", WS_VISIBLE | WS_CHILD,
            leftCol + 90, y, 25, 25, hwnd, (HMENU)ID_AREA_WIDTH_DEC, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+", WS_VISIBLE | WS_CHILD,
            leftCol + 120, y, 25, 25, hwnd, (HMENU)ID_AREA_WIDTH_INC, nullptr, nullptr);
        y += 35;


        CreateWindow(L"STATIC", L"Area height:", WS_VISIBLE | WS_CHILD,
            leftCol, y, 100, 20, hwnd, nullptr, nullptr, nullptr);
        y += 25;
        CreateWindow(L"STATIC", (std::to_wstring(config.areaHeight) + L" mm").c_str(),
            WS_VISIBLE | WS_CHILD | SS_CENTER,
            leftCol + 20, y, 60, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"-", WS_VISIBLE | WS_CHILD,
            leftCol + 90, y, 25, 25, hwnd, (HMENU)ID_AREA_HEIGHT_DEC, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"+", WS_VISIBLE | WS_CHILD,
            leftCol + 120, y, 25, 25, hwnd, (HMENU)ID_AREA_HEIGHT_INC, nullptr, nullptr);
        y += 40;


        CreateWindow(L"STATIC", L"═══ SETTINGS ═══", WS_VISIBLE | WS_CHILD | SS_CENTER,
            rightCol, 20, 200, 25, hwnd, nullptr, nullptr, nullptr);

        int rightY = 55;


        HWND startStopBtn = CreateWindow(L"BUTTON", running ? L"STOP" : L"START",
            WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
            rightCol, rightY, 180, 40, hwnd, (HMENU)ID_START_STOP, nullptr, nullptr);


        HFONT bigFont = CreateFont(16, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
        SendMessage(startStopBtn, WM_SETFONT, (WPARAM)bigFont, TRUE);
        rightY += 55;


        CreateWindow(L"STATIC", L"Rotation:", WS_VISIBLE | WS_CHILD,
            rightCol, rightY, 80, 20, hwnd, nullptr, nullptr, nullptr);
        rightY += 25;
        CreateWindow(L"STATIC", GetRotationText().c_str(), WS_VISIBLE | WS_CHILD | SS_CENTER,
            rightCol + 20, rightY, 100, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"Change", WS_VISIBLE | WS_CHILD,
            rightCol + 130, rightY, 50, 25, hwnd, (HMENU)ID_ROTATION_TOGGLE, nullptr, nullptr);
        rightY += 35;


        if (monitors.size() > 1) {
            CreateWindow(L"STATIC", L"Monitor:", WS_VISIBLE | WS_CHILD,
                rightCol, rightY, 80, 20, hwnd, nullptr, nullptr, nullptr);
            rightY += 25;
            CreateWindow(L"STATIC", (std::to_wstring(config.currentMonitor + 1) + L"/" + std::to_wstring(monitors.size())).c_str(),
                WS_VISIBLE | WS_CHILD | SS_CENTER, rightCol + 20, rightY, 60, 20, hwnd, nullptr, nullptr, nullptr);
            CreateWindow(L"BUTTON", L"Switch", WS_VISIBLE | WS_CHILD,
                rightCol + 90, rightY, 60, 25, hwnd, (HMENU)ID_MONITOR_SWITCH, nullptr, nullptr);
            rightY += 40;
        }


        CreateWindow(L"STATIC", L"═══ ADVANCED ═══", WS_VISIBLE | WS_CHILD | SS_CENTER,
            rightCol, rightY, 200, 25, hwnd, nullptr, nullptr, nullptr);
        rightY += 35;


        CreateWindow(L"STATIC", L"Prediction:", WS_VISIBLE | WS_CHILD,
            rightCol, rightY, 120, 20, hwnd, nullptr, nullptr, nullptr);
        rightY += 25;
        CreateWindow(L"STATIC", config.movementPrediction ? L"ON" : L"OFF",
            WS_VISIBLE | WS_CHILD | SS_CENTER,
            rightCol + 20, rightY, 80, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"Switch", WS_VISIBLE | WS_CHILD,
            rightCol + 110, rightY, 60, 25, hwnd, (HMENU)ID_PREDICTION_TOGGLE, nullptr, nullptr);
        rightY += 30;

        if (config.movementPrediction) {
            CreateWindow(L"STATIC", L"Strength:", WS_VISIBLE | WS_CHILD,
                rightCol + 10, rightY, 80, 20, hwnd, nullptr, nullptr, nullptr);
            rightY += 25;
            CreateWindow(L"STATIC", std::to_wstring(config.predictionStrength).c_str(),
                WS_VISIBLE | WS_CHILD | SS_CENTER,
                rightCol + 30, rightY, 40, 20, hwnd, nullptr, nullptr, nullptr);
            CreateWindow(L"BUTTON", L"-", WS_VISIBLE | WS_CHILD,
                rightCol + 80, rightY, 25, 25, hwnd, (HMENU)ID_PREDICTION_STR_DEC, nullptr, nullptr);
            CreateWindow(L"BUTTON", L"+", WS_VISIBLE | WS_CHILD,
                rightCol + 110, rightY, 25, 25, hwnd, (HMENU)ID_PREDICTION_STR_INC, nullptr, nullptr);
            rightY += 35;
        }

        CreateWindow(L"STATIC", L"Smoothing:", WS_VISIBLE | WS_CHILD,
            rightCol, rightY, 120, 20, hwnd, nullptr, nullptr, nullptr);
        rightY += 25;
        CreateWindow(L"STATIC", config.smoothingEnabled ? L"ON" : L"OFF",
            WS_VISIBLE | WS_CHILD | SS_CENTER,
            rightCol + 20, rightY, 80, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"Switch", WS_VISIBLE | WS_CHILD,
            rightCol + 110, rightY, 60, 25, hwnd, (HMENU)ID_SMOOTHING_TOGGLE, nullptr, nullptr);
        rightY += 30;

        if (config.smoothingEnabled) {
            CreateWindow(L"STATIC", L"Strength:", WS_VISIBLE | WS_CHILD,
                rightCol + 10, rightY, 80, 20, hwnd, nullptr, nullptr, nullptr);
            rightY += 25;
            CreateWindow(L"STATIC", std::to_wstring(config.smoothingStrength).c_str(),
                WS_VISIBLE | WS_CHILD | SS_CENTER,
                rightCol + 30, rightY, 40, 20, hwnd, nullptr, nullptr, nullptr);
            CreateWindow(L"BUTTON", L"-", WS_VISIBLE | WS_CHILD,
                rightCol + 80, rightY, 25, 25, hwnd, (HMENU)ID_SMOOTHING_STR_DEC, nullptr, nullptr);
            CreateWindow(L"BUTTON", L"+", WS_VISIBLE | WS_CHILD,
                rightCol + 110, rightY, 25, 25, hwnd, (HMENU)ID_SMOOTHING_STR_INC, nullptr, nullptr);
            rightY += 35;
        }


        CreateWindow(L"STATIC", L"Clicks:", WS_VISIBLE | WS_CHILD,
            rightCol, rightY, 120, 20, hwnd, nullptr, nullptr, nullptr);
        rightY += 25;
        CreateWindow(L"STATIC", config.clickEnabled ? L"ON" : L"OFF",
            WS_VISIBLE | WS_CHILD | SS_CENTER,
            rightCol + 20, rightY, 80, 20, hwnd, nullptr, nullptr, nullptr);
        CreateWindow(L"BUTTON", L"Switch", WS_VISIBLE | WS_CHILD,
            rightCol + 110, rightY, 60, 25, hwnd, (HMENU)ID_CLICK_TOGGLE, nullptr, nullptr);
        rightY += 35;
    }

    std::wstring GetRotationText() {
        switch (config.rotation) {
        case 0: return L"0°";
        case 1: return L"90°";
        case 2: return L"180°";
        case 3: return L"270°";
        default: return L"0°";
        }
    }

    void DrawTabletArea(HWND hwnd) {
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


        float areaWidthRatio, areaHeightRatio;


        if (config.rotation == 0 || config.rotation == 2) {
            areaWidthRatio = (float)config.areaHeight / currentTablet.widthMM;
            areaHeightRatio = (float)config.areaWidth / currentTablet.heightMM;
        }
        else {
            areaWidthRatio = (float)config.areaWidth / currentTablet.widthMM;
            areaHeightRatio = (float)config.areaHeight / currentTablet.heightMM;
        }

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


    void UpdateVelocityHistory(const Vec2f& newPosition, std::chrono::high_resolution_clock::time_point timestamp) {
        if (!hasLastPosition.load()) {
            lastPositionTime = timestamp;
            return;
        }

        Vec2f lastPos = lastScreenPos.load();
        lastPos.Prefetch();

        auto timeDiff = std::chrono::duration_cast<std::chrono::microseconds>(timestamp - lastPositionTime);
        float deltaTime = timeDiff.count() / 1000000.0f;

        if (deltaTime > 0.0f && deltaTime < 0.1f) {
            __m128 vnewPos = _mm_set_ps(0.0f, 0.0f, newPosition.y, newPosition.x);
            __m128 vlastPos = _mm_set_ps(0.0f, 0.0f, lastPos.y, lastPos.x);
            __m128 vdelta = _mm_sub_ps(vnewPos, vlastPos);
            __m128 vdeltaTime = _mm_set1_ps(1.0f / deltaTime);
            __m128 vvelocity = _mm_mul_ps(vdelta, vdeltaTime);

            float velocityData[4];
            _mm_store_ps(velocityData, vvelocity);

            uint64_t timestampMicros = std::chrono::duration_cast<std::chrono::microseconds>(
                timestamp.time_since_epoch()).count();

            velocityHistory.AddSample(velocityData[0], velocityData[1], timestampMicros);
        }

        secondLastPositionTime = lastPositionTime;
        lastPositionTime = timestamp;
    }

    Vec2f CalculateSmoothedVelocity() {
        return velocityHistory.CalculateWeightedAverage();
    }

    Vec2i ApplyRawDataSmoothing(const Vec2i& newRawPos, bool inProximity, bool isTouching) {
        if (!config.smoothingEnabled) {
            return newRawPos;
        }

        if (!inProximity && !isTouching) {
            hasSmoothedData.store(false);
            return newRawPos;
        }

        if (!hasSmoothedData.load()) {
            smoothedRawPos.store(newRawPos);
            hasSmoothedData.store(true);
            return newRawPos;
        }

        Vec2i currentSmoothed = smoothedRawPos.load();

        float deltaX = (float)(newRawPos.x - currentSmoothed.x);
        float deltaY = (float)(newRawPos.y - currentSmoothed.y);
        float distance = sqrt(deltaX * deltaX + deltaY * deltaY);

        if (distance < SMOOTHING_THRESHOLD_BASE) {
            return currentSmoothed;
        }

        float smoothingFactor = 1.0f - (config.smoothingStrength * 0.04f);
        smoothingFactor = max(0.1f, min(0.98f, smoothingFactor));

        Vec2i result;
        result.x = (int)(currentSmoothed.x + (deltaX * smoothingFactor));
        result.y = (int)(currentSmoothed.y + (deltaY * smoothingFactor));

        result.x = max(0, min(currentTablet.maxX, result.x));
        result.y = max(0, min(currentTablet.maxY, result.y));

        smoothedRawPos.store(result);
        return result;
    }
    void HandleCommand(int commandId) {
        switch (commandId) {

        case ID_CENTER_UP: MoveAreaCenter(0, -1); break;
        case ID_CENTER_DOWN: MoveAreaCenter(0, 1); break;
        case ID_CENTER_LEFT: MoveAreaCenter(-1, 0); break;
        case ID_CENTER_RIGHT: MoveAreaCenter(1, 0); break;

        case ID_AREA_WIDTH_DEC: AdjustAreaSize(-1, 0); break;
        case ID_AREA_WIDTH_INC: AdjustAreaSize(1, 0); break;
        case ID_AREA_HEIGHT_DEC: AdjustAreaSize(0, -1); break;
        case ID_AREA_HEIGHT_INC: AdjustAreaSize(0, 1); break;

        case ID_ROTATION_TOGGLE: ConfigureRotation(); break;
        case ID_MONITOR_SWITCH: SwitchMonitor(); break;
        case ID_START_STOP:
            if (running) Stop();
            else Start();
            break;

        case ID_PREDICTION_TOGGLE: TogglePrediction(); break;
        case ID_PREDICTION_STR_DEC: AdjustPredictionStrength(-1); break;
        case ID_PREDICTION_STR_INC: AdjustPredictionStrength(1); break;
        case ID_CLICK_TOGGLE: ToggleClick(); break;

        case ID_SMOOTHING_TOGGLE: ToggleSmoothing(); break;
        case ID_SMOOTHING_STR_DEC: AdjustSmoothingStrength(-1); break;
        case ID_SMOOTHING_STR_INC: AdjustSmoothingStrength(1); break;
        }

        UpdateGUI();
    }

    void UpdateGUI() {

        if (visualArea) {
            InvalidateRect(visualArea, nullptr, FALSE);
        }


        if (mainWindow) {

            RECT windowRect;
            GetWindowRect(mainWindow, &windowRect);


            EnumChildWindows(mainWindow, [](HWND hwnd, LPARAM lParam) -> BOOL {
                HWND visualArea = reinterpret_cast<HWND>(lParam);
                if (hwnd != visualArea) {
                    DestroyWindow(hwnd);
                }
                return TRUE;
                }, reinterpret_cast<LPARAM>(visualArea));


            CreateControls(mainWindow);


            UpdateWindow(mainWindow);
        }
    }

private:
    bool hasSSE, hasSSE2, hasAVX;

    HANDLE consoleHandle;
    COORD loggingLinePos;

    alignas(16) std::atomic<Vec2f> currentVelocity;
    std::atomic<double> predictionAmount{ 0.0 };

    std::atomic<int> currentTabletX{ 0 };
    std::atomic<int> currentTabletY{ 0 };
    std::atomic<int> currentScreenX{ 0 };
    std::atomic<int> currentScreenY{ 0 };
    std::atomic<bool> isCurrentlyClicking{ false };


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

    alignas(16) std::atomic<Vec2f> lastScreenPos;
    std::atomic<bool> hasLastPosition{ false };
    std::atomic<bool> wasInProximityLast{ false };

    std::atomic<bool> lastTouchState{ false };
    std::atomic<bool> currentlyPressed{ false };

    std::atomic<bool> emergencyShutdown{ false };

    alignas(16) std::atomic<Vec2i> lastValidRawPos;
    std::atomic<bool> hasValidData{ false };

    const TabletSpec TABLET_SPECS[4] = {
        {0, 0, 0, 0, TabletType::UNKNOWN, "Unknown"},
        {21610, 13498, 216, 135, TabletType::WACOM_CTL672, "Wacom CTL-672"},
        {30480, 20320, 152, 102, TabletType::XPPEN_STAR_G640, "XPPen Star G 640"},
        {15200, 9500, 152, 95, TabletType::WACOM_CTL472, "Wacom CTL-472"}
    };


    __forceinline bool ParseTabletDataZeroCopy(BYTE* buffer, DWORD length, TabletData& data) {
        data.timestamp = std::chrono::high_resolution_clock::now();
        data.inProximity = false;
        data.isTouching = false;
        data.rawPos = Vec2i(0, 0);
        data.isValid = false;

        int rawX, rawY;
        switch (currentTablet.type) {
        case TabletType::WACOM_CTL672:
            if (!ParseWacomDataZeroCopy(buffer, length, rawX, rawY, data.inProximity, data.isTouching)) return false;
            break;
        case TabletType::WACOM_CTL472:
            if (!ParseWacomCTL472DataZeroCopy(buffer, length, rawX, rawY, data.inProximity, data.isTouching)) return false;
            break;
        case TabletType::XPPEN_STAR_G640:
            if (!ParseXPPenDataZeroCopy(buffer, length, rawX, rawY, data.inProximity, data.isTouching)) return false;
            break;
        default:
            return false;
        }

        data.rawPos = Vec2i(rawX, rawY);
        data.isValid = IsValidTabletDataFast(data.rawPos, data.inProximity);
        return data.isValid;
    }

    __forceinline void UpdateFastAccessDataZeroCopy(const TabletData& data) {

        memcpy(&fastAccessData, &data, sizeof(TabletData));
        _mm_sfence();
        fastDataValid = true;
    }

    void ProcessBatchedReports() {
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

    void ProcessTabletDataBatch(TabletData* batch, size_t count) {

        for (size_t i = 0; i < count; i++) {
            if (batch[i].isValid) {
                tabletDataBuffer.TryPush(batch[i]);
            }
        }
    }

    Vec2f CalculateScreenPositionFast(const TabletData& data) {

        float normalizedX = (data.rawPos.x - tabletLookup.centerOffsetX) * tabletLookup.scaleX;
        float normalizedY = (data.rawPos.y - tabletLookup.centerOffsetY) * tabletLookup.scaleY;


        if (normalizedX < 0.0f || normalizedX > 1.0f || normalizedY < 0.0f || normalizedY > 1.0f) {
            return Vec2f(-1.0f, -1.0f);
        }

        const Monitor& monitor = monitors[config.currentMonitor];
        return Vec2f(
            monitor.x + normalizedX * monitor.width,
            monitor.y + normalizedY * monitor.height
        );
    }

    __forceinline bool IsValidTabletDataFast(const Vec2i& rawPos, bool inProximity) {

        return rawPos.x >= 0 && rawPos.y >= 0 &&
            rawPos.x <= tabletLookup.maxX && rawPos.y <= tabletLookup.maxY &&
            !(rawPos.x == 0 && rawPos.y == 0 && !inProximity);
    }


    __forceinline bool ParseWacomDataZeroCopy(BYTE* data, DWORD length, int& rawX, int& rawY, bool& inProximity, bool& isTouching) {
        if (length < 8) return false;

        if (data[0] == 0x01) {
            rawX = data[2] | (data[3] << 8);
            rawY = data[4] | (data[5] << 8);
            inProximity = (data[1] & 0x01) != 0;
            isTouching = (data[1] & 0x02) != 0;
            return true;
        }
        else if (data[0] == 0x02) {
            rawX = data[2] | (data[3] << 8);
            rawY = data[4] | (data[5] << 8);
            inProximity = (data[1] & 0x20) != 0;
            isTouching = (data[1] & 0x01) != 0;
            return true;
        }
        return false;
    }

    __forceinline bool ParseWacomCTL472DataZeroCopy(BYTE* data, DWORD length, int& rawX, int& rawY, bool& inProximity, bool& isTouching) {
        if (length < 10) return false;

        if (data[0] == 0x01 || data[0] == 0x02) {
            rawX = data[2] | (data[3] << 8);
            rawY = data[4] | (data[5] << 8);
            inProximity = (data[1] & 0x20) != 0;
            isTouching = (data[1] & 0x01) != 0;
            return true;
        }
        return false;
    }

    __forceinline bool ParseXPPenDataZeroCopy(BYTE* data, DWORD length, int& rawX, int& rawY, bool& inProximity, bool& isTouching) {
        if (length < 14) return false;

        if (data[0] == 0x02 && (data[1] == 0xA0 || data[1] == 0xA1)) {
            rawX = data[2] | (data[3] << 8);
            rawY = data[4] | (data[5] << 8);
            inProximity = true;
            isTouching = (data[1] == 0xA1);
            return true;
        }
        return false;
    }

public:
    HighPerformanceTabletDriver() : deviceHandle(INVALID_HANDLE_VALUE) {
        hasSSE = SIMDMath::HasSSE();
        hasSSE2 = SIMDMath::HasSSE2();
        hasAVX = SIMDMath::HasAVX();

        std::cout << "SIMD Support: SSE=" << (hasSSE ? "YES" : "NO")
            << ", SSE2=" << (hasSSE2 ? "YES" : "NO")
            << ", AVX=" << (hasAVX ? "YES" : "NO") << std::endl;

        InitializeCPUAffinity();

        currentTablet = TABLET_SPECS[0];
        DetectMonitors();
        LoadConfig();

        consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
        loggingLinePos = { 0, 0 };

        Vec2f zero(0.0f, 0.0f);
        Vec2i zeroI(0, 0);
        currentVelocity.store(zero);
        lastScreenPos.store(zero);
        lastValidRawPos.store(zeroI);

        fastDataValid = false;
        memset(&fastAccessData, 0, sizeof(TabletData));

        _mm_prefetch(reinterpret_cast<const char*>(&velocityHistory), _MM_HINT_T1);
        _mm_prefetch(reinterpret_cast<const char*>(&fastAccessData), _MM_HINT_T1);
        _mm_prefetch(reinterpret_cast<const char*>(&tabletDataBuffer), _MM_HINT_T1);

        memset(workingPositions, 0, sizeof(workingPositions));
        memset(workingWeights, 0, sizeof(workingWeights));
        memset(workingTimestamps, 0, sizeof(workingTimestamps));

        Vec2i zeroSmoothed(0, 0);
        smoothedRawPos.store(zeroSmoothed);
        hasSmoothedData.store(false);
    }

    void InitializeTabletLookupTable() {
        if (currentTablet.type == TabletType::UNKNOWN) return;

        Vec2i areaMin, areaSize;
        CalculateAreaBounds(areaMin, areaSize);

        tabletLookup.maxX = currentTablet.maxX;
        tabletLookup.maxY = currentTablet.maxY;
        tabletLookup.scaleX = 1.0f / areaSize.x;
        tabletLookup.scaleY = 1.0f / areaSize.y;
        tabletLookup.centerOffsetX = areaMin.x;
        tabletLookup.centerOffsetY = areaMin.y;


        _mm_prefetch(reinterpret_cast<const char*>(&tabletLookup), _MM_HINT_T0);
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

    void LoadConfig() {
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
                    else if (key == "clickEnabled") config.clickEnabled = (value == "1");
                    else if (key == "currentMonitor") config.currentMonitor = std::stoi(value);
                    else if (key == "smoothingEnabled") config.smoothingEnabled = (value == "1");
                    else if (key == "smoothingStrength") config.smoothingStrength = std::stoi(value);
                }
            }
            file.close();
        }
    }

    void SaveConfig() {
        std::ofstream file("driver.config");
        if (file.is_open()) {
            file << "areaWidth = " << config.areaWidth << std::endl;
            file << "areaHeight = " << config.areaHeight << std::endl;
            file << "areaCenterX = " << config.areaCenterX << std::endl;
            file << "areaCenterY = " << config.areaCenterY << std::endl;
            file << "rotation = " << config.rotation << std::endl;
            file << "movementPrediction = " << (config.movementPrediction ? 1 : 0) << std::endl;
            file << "predictionStrength = " << config.predictionStrength << std::endl;
            file << "clickEnabled = " << (config.clickEnabled ? 1 : 0) << std::endl;
            file << "currentMonitor = " << config.currentMonitor << std::endl;
            file << "smoothingEnabled = " << (config.smoothingEnabled ? 1 : 0) << std::endl;
            file << "smoothingStrength = " << config.smoothingStrength << std::endl;
            file.close();
        }
    }

    bool Initialize() {
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
                            InitializeTabletLookupTable();

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

private:
    void SetThreadSafePriority() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
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

public:
    void MoveAreaCenter(int deltaX, int deltaY) {
        config.areaCenterX += deltaX;
        config.areaCenterY += deltaY;

        config.areaCenterX = max(config.areaWidth / 2, min(currentTablet.widthMM - config.areaWidth / 2, config.areaCenterX));
        config.areaCenterY = max(config.areaHeight / 2, min(currentTablet.heightMM - config.areaHeight / 2, config.areaCenterY));
        SaveConfig();
    }

    void AdjustAreaSize(int widthDelta, int heightDelta) {
        config.areaWidth = max(5, min(currentTablet.widthMM, config.areaWidth + widthDelta));
        config.areaHeight = max(5, min(currentTablet.heightMM, config.areaHeight + heightDelta));

        config.areaCenterX = max(config.areaWidth / 2, min(currentTablet.widthMM - config.areaWidth / 2, config.areaCenterX));
        config.areaCenterY = max(config.areaHeight / 2, min(currentTablet.heightMM - config.areaHeight / 2, config.areaCenterY));
        SaveConfig();
    }

    void ConfigureRotation() {
        config.rotation = (config.rotation + 1) % 4;
        SaveConfig();
    }

    void SwitchMonitor() {
        if (monitors.size() > 1) {
            config.currentMonitor = (config.currentMonitor + 1) % monitors.size();
            SaveConfig();
        }
    }

    void TogglePrediction() {
        config.movementPrediction = !config.movementPrediction;
        SaveConfig();
    }

    void ToggleSmoothing() {
        config.smoothingEnabled = !config.smoothingEnabled;
        if (!config.smoothingEnabled) {
            hasSmoothedData.store(false);
        }
        SaveConfig();
    }

    void AdjustSmoothingStrength(int delta) {
        config.smoothingStrength = max(1, min(20, config.smoothingStrength + delta));
        SaveConfig();
    }

    void ToggleClick() {
        config.clickEnabled = !config.clickEnabled;
        SaveConfig();
    }

    void AdjustPredictionStrength(int delta) {
        config.predictionStrength = max(1, min(20, config.predictionStrength + delta));
        SaveConfig();
    }

    void RestartDriver() {
        if (running) {
            Stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        Start();
    }

    void RunConfiguration() {

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
            L"Ultra Tablet Driver v4.1 - GUI",
            WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
            CW_USEDEFAULT, CW_USEDEFAULT, 520, 650,
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

    void Start() {
        if (deviceHandle == INVALID_HANDLE_VALUE) {
            std::cerr << "Device not initialized" << std::endl;
            return;
        }

        emergencyShutdown = false;
        running = true;
        hasLastPosition = false;
        wasInProximityLast = false;
        lastTouchState = false;
        currentlyPressed = false;

        hasValidData = false;
        Vec2i zeroPos(-1, -1);
        lastValidRawPos.store(zeroPos);

        hasSmoothedData.store(false);
        Vec2i zeroSmoothed(0, 0);
        smoothedRawPos.store(zeroSmoothed);

        velocityHistory.Clear();

        inputThread = std::thread(&HighPerformanceTabletDriver::SafeInputLoop, this);
        processingThread = std::thread(&HighPerformanceTabletDriver::SafeProcessingLoop, this);
    }

    void Stop() {
        emergencyShutdown = true;
        running = false;
        guiRunning = false;

        if (currentlyPressed && config.clickEnabled) {
            INPUT input = {};
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
            currentlyPressed = false;
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

private:
    void SafeInputLoop() {
        SetOptimalThreadAffinity(0);

        BYTE buffer[64];
        DWORD bytesRead;

        while (running && !emergencyShutdown) {
            try {
                if (ReadFile(deviceHandle, buffer, sizeof(buffer), &bytesRead, NULL)) {
                    ProcessRawTabletData(buffer, bytesRead);
                }
            }
            catch (...) {
                emergencyShutdown = true;
                break;
            }
        }
    }

    void SafeProcessingLoop() {
        SetOptimalThreadAffinity(1);


        TabletData localData;
        Vec2f baseScreenPos, finalScreenPos;

        while (running && !emergencyShutdown) {
            try {


                if (fastDataValid) {
                    localData = fastAccessData;
                    _mm_lfence();

                    if (localData.isValid) {
                        if (CalculateScreenPosition(localData, baseScreenPos)) {
                            ApplyPrediction(baseScreenPos, finalScreenPos, localData.timestamp);

                            MoveCursorToPosition(finalScreenPos);

                            HandleMouseClick(localData.isTouching);
                        }
                    }
                    fastDataValid = false;
                }


                if (!batchBuffer.IsEmpty()) {
                    ProcessBatchedReports();
                }
                while (tabletDataBuffer.TryPop(localData)) {
                    if (localData.isValid) {
                        if (CalculateScreenPosition(localData, baseScreenPos)) {
                            ApplyPrediction(baseScreenPos, finalScreenPos, localData.timestamp);

                            MoveCursorToPosition(finalScreenPos);

                            HandleMouseClick(localData.isTouching);
                        }
                    }
                }


                if (tabletDataBuffer.IsEmpty()) {
                    _mm_pause();
                }

            }
            catch (...) {
                emergencyShutdown = true;
                break;
            }
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

    bool CalculateScreenPosition(const TabletData& data, Vec2f& screenPos) {
        bool wasJustLifted = wasInProximityLast.load() && !data.inProximity;
        wasInProximityLast.store(data.inProximity);

        if (!data.inProximity && !data.isTouching) {
            hasLastPosition = false;

            velocityHistory.Clear();
            return false;
        }

        Vec2i areaMin, areaSize;
        CalculateAreaBounds(areaMin, areaSize);

        const Monitor& currentMonitor = monitors[config.currentMonitor];

        currentTabletX.store(data.rawPos.x);
        currentTabletY.store(data.rawPos.y);
        isCurrentlyClicking.store(data.isTouching);

        if (!IsWithinArea(data.rawPos, areaMin, areaSize)) {
            MoveToNearestEdge(data.rawPos, areaMin, areaSize, currentMonitor);

            if (config.clickEnabled && currentlyPressed) {
                HandleMouseClick(false);
            }
            hasLastPosition = false;

            velocityHistory.Clear();
            return false;
        }

        Vec2f normalizedPos = SIMDMath::NormalizeCoords(data.rawPos, areaMin, areaSize);
        Vec2f rotatedPos = ApplyRotationToNormalized(normalizedPos, config.rotation);

        Vec2i screenSize, screenOffset;
        GetRotatedScreenBounds(currentMonitor, screenSize, screenOffset);

        Vec2i screenPosInt = SIMDMath::MapToScreen(rotatedPos, screenSize, screenOffset);
        screenPos = Vec2f((float)screenPosInt.x, (float)screenPosInt.y);

        return true;
    }

    Vec2f ApplyRotationToNormalized(const Vec2f& normalizedPos, int rotation) {
        Vec2f result;

        switch (rotation) {
        case 0:
            result = normalizedPos;
            break;
        case 1:
            result.x = 1.0f - normalizedPos.y;
            result.y = normalizedPos.x;
            break;
        case 2:
            result.x = 1.0f - normalizedPos.x;
            result.y = 1.0f - normalizedPos.y;
            break;
        case 3:
            result.x = normalizedPos.y;
            result.y = 1.0f - normalizedPos.x;
            break;
        default:
            result = normalizedPos;
            break;
        }

        return result;
    }

    void GetRotatedScreenBounds(const Monitor& monitor, Vec2i& screenSize, Vec2i& screenOffset) {
        screenOffset = Vec2i(monitor.x, monitor.y);
        screenSize = Vec2i(monitor.width, monitor.height);
    }

    void ApplyPrediction(const Vec2f& baseScreenPos, Vec2f& finalScreenPos,
        std::chrono::high_resolution_clock::time_point timestamp) {

        finalScreenPos = baseScreenPos;


        UpdateVelocityHistory(baseScreenPos, timestamp);

        if (config.movementPrediction && hasLastPosition.load()) {
            Vec2f smoothedVelocity = CalculateSmoothedVelocity();


            float velocityMagnitude = SIMDMath::Length(smoothedVelocity);


            if (velocityMagnitude > 50.0f) {

                float predictionTime = 0.012f;


                float adaptiveFactor = min(velocityMagnitude / 1000.0f, 1.0f);
                float finalPredictionTime = predictionTime * adaptiveFactor * (config.predictionStrength * 0.1f);


                Vec2f prediction = SIMDMath::Scale(smoothedVelocity, finalPredictionTime);
                finalScreenPos = SIMDMath::Add(baseScreenPos, prediction);


                currentVelocity.store(smoothedVelocity);
                predictionAmount.store(SIMDMath::Length(prediction));
            }
            else {

                Vec2f zeroVel(0.0f, 0.0f);
                currentVelocity.store(zeroVel);
                predictionAmount.store(0.0);
            }
        }
        else {
            Vec2f zeroVel(0.0f, 0.0f);
            currentVelocity.store(zeroVel);
            predictionAmount.store(0.0);
        }


        const Monitor& currentMonitor = monitors[config.currentMonitor];

        int safeMargin = 5;
        finalScreenPos.x = max((float)(currentMonitor.x + safeMargin),
            min((float)(currentMonitor.x + currentMonitor.width - safeMargin), finalScreenPos.x));
        finalScreenPos.y = max((float)(currentMonitor.y + safeMargin),
            min((float)(currentMonitor.y + currentMonitor.height - safeMargin), finalScreenPos.y));

        lastScreenPos.store(baseScreenPos);
        hasLastPosition = true;
    }

    void MoveCursorToPosition(const Vec2f& screenPos) {
        int screenX = (int)screenPos.x;
        int screenY = (int)screenPos.y;

        const Monitor& currentMonitor = monitors[config.currentMonitor];

        if ((screenX <= currentMonitor.x + 10 && screenY <= currentMonitor.y + 10) ||
            (screenX >= currentMonitor.x + currentMonitor.width - 10 &&
                screenY <= currentMonitor.y + 10)) {
            return;
        }

        try {
            POINT currentPos;
            GetCursorPos(&currentPos);

            int deltaX = screenX - currentPos.x;
            int deltaY = screenY - currentPos.y;

            if (abs(deltaX) < 1 && abs(deltaY) < 1) {
                return;
            }

            INPUT input = {};
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_MOVE;
            input.mi.dx = deltaX;
            input.mi.dy = deltaY;
            SendInput(1, &input, sizeof(INPUT));

            currentScreenX.store(screenX);
            currentScreenY.store(screenY);
        }
        catch (...) {
        }
    }

    void ProcessRawTabletData(BYTE* buffer, DWORD length) {
        if (length < 6 || buffer == nullptr) return;

        TabletData localData;
        localData.timestamp = std::chrono::high_resolution_clock::now();
        localData.inProximity = false;
        localData.isTouching = false;
        localData.rawPos = Vec2i(0, 0);
        localData.isValid = false;

        try {
            int rawX, rawY;
            switch (currentTablet.type) {
            case TabletType::WACOM_CTL672:
                ParseWacomData(buffer, length, rawX, rawY, localData.inProximity, localData.isTouching);
                break;
            case TabletType::WACOM_CTL472:
                ParseWacomCTL472Data(buffer, length, rawX, rawY, localData.inProximity, localData.isTouching);
                break;
            case TabletType::XPPEN_STAR_G640:
                ParseXPPenData(buffer, length, rawX, rawY, localData.inProximity, localData.isTouching);
                break;
            default:
                return;
            }

            Vec2i originalRawPos(rawX, rawY);
            originalRawPos.Prefetch();

            Vec2i smoothedPos = ApplyRawDataSmoothing(originalRawPos, localData.inProximity, localData.isTouching);

            localData.rawPos = smoothedPos;

        }
        catch (...) {
            return;
        }

        if (!IsValidTabletData(localData.rawPos, localData.inProximity)) {
            return;
        }

        localData.isValid = true;

        if (localData.inProximity || localData.isTouching) {
            fastAccessData = localData;
            _mm_sfence();
            fastDataValid = true;

            if (!batchBuffer.TryPush(localData)) {
                ProcessBatchedReports();
                tabletDataBuffer.TryPush(localData);
            }
        }
    }

    Vec2f CubicBezier(const Vec2f& p0, const Vec2f& p1, const Vec2f& p2, const Vec2f& p3, float t) {
        float u = 1.0f - t;
        float tt = t * t;
        float uu = u * u;
        float uuu = uu * u;
        float ttt = tt * t;

        Vec2f result = SIMDMath::Scale(p0, uuu);
        result = SIMDMath::Add(result, SIMDMath::Scale(p1, 3.0f * uu * t));
        result = SIMDMath::Add(result, SIMDMath::Scale(p2, 3.0f * u * tt));
        result = SIMDMath::Add(result, SIMDMath::Scale(p3, ttt));

        return result;
    }

    bool IsValidTabletData(const Vec2i& rawPos, bool inProximity) {
        if (rawPos.x < 0 || rawPos.y < 0 || rawPos.x > currentTablet.maxX || rawPos.y > currentTablet.maxY) {
            return false;
        }

        if ((rawPos.x == 0 && rawPos.y == 0) && !inProximity) {
            return false;
        }

        if (rawPos.x == 0xFFFF || rawPos.y == 0xFFFF || rawPos.x == 0x7FFF || rawPos.y == 0x7FFF) {
            return false;
        }

        if (hasValidData.load()) {
            Vec2i lastPos = lastValidRawPos.load();

            if (lastPos.x >= 0 && lastPos.y >= 0) {
                Vec2i delta = Vec2i(abs(rawPos.x - lastPos.x), abs(rawPos.y - lastPos.y));

                int maxJumpX = (currentTablet.maxX * 80) / 100;
                int maxJumpY = (currentTablet.maxY * 80) / 100;

                if (delta.x > maxJumpX || delta.y > maxJumpY) {
                    return false;
                }
            }
        }

        lastValidRawPos.store(rawPos);
        hasValidData.store(true);

        return true;
    }

    void MoveToNearestEdge(const Vec2i& rawPos, const Vec2i& areaMin, const Vec2i& areaSize, const Monitor& currentMonitor) {
        Vec2i areaMax = Vec2i(areaMin.x + areaSize.x, areaMin.y + areaSize.y);
        Vec2i nearestPos = Vec2i(
            max(areaMin.x, min(areaMax.x, rawPos.x)),
            max(areaMin.y, min(areaMax.y, rawPos.y))
        );

        Vec2f normalizedPos = SIMDMath::NormalizeCoords(nearestPos, areaMin, areaSize);
        Vec2f rotatedPos = ApplyRotationToNormalized(normalizedPos, config.rotation);

        Vec2i screenSize, screenOffset;
        GetRotatedScreenBounds(currentMonitor, screenSize, screenOffset);
        Vec2i screenPos = SIMDMath::MapToScreen(rotatedPos, screenSize, screenOffset);

        screenPos.x = max(currentMonitor.x, min(currentMonitor.x + currentMonitor.width - 1, screenPos.x));
        screenPos.y = max(currentMonitor.y, min(currentMonitor.y + currentMonitor.height - 1, screenPos.y));

        try {
            POINT currentPos;
            GetCursorPos(&currentPos);

            int deltaX = screenPos.x - currentPos.x;
            int deltaY = screenPos.y - currentPos.y;

            INPUT input = {};
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_MOVE;
            input.mi.dx = deltaX;
            input.mi.dy = deltaY;
            SendInput(1, &input, sizeof(INPUT));
        }
        catch (...) {
        }
    }

    void CalculateAreaBounds(Vec2i& areaMin, Vec2i& areaSize) const {
        Vec2i center(
            config.areaCenterX * currentTablet.maxX / currentTablet.widthMM,
            config.areaCenterY * currentTablet.maxY / currentTablet.heightMM
        );

        Vec2i halfSize;


        if (config.rotation == 0 || config.rotation == 2) {
            halfSize = Vec2i(
                (config.areaHeight * currentTablet.maxX / currentTablet.widthMM) / 2,
                (config.areaWidth * currentTablet.maxY / currentTablet.heightMM) / 2
            );
        }
        else {

            halfSize = Vec2i(
                (config.areaWidth * currentTablet.maxX / currentTablet.widthMM) / 2,
                (config.areaHeight * currentTablet.maxY / currentTablet.heightMM) / 2
            );
        }

        areaMin = Vec2i(center.x - halfSize.x, center.y - halfSize.y);
        areaSize = Vec2i(halfSize.x * 2, halfSize.y * 2);
    }

    bool IsWithinArea(const Vec2i& pos, const Vec2i& areaMin, const Vec2i& areaSize) const {
        return pos.x >= areaMin.x && pos.x <= (areaMin.x + areaSize.x) &&
            pos.y >= areaMin.y && pos.y <= (areaMin.y + areaSize.y);
    }

    void HandleMouseClick(bool isTouching) {
        if (!config.clickEnabled) return;

        bool lastTouch = lastTouchState.load();
        if (isTouching != lastTouch) {
            try {
                INPUT input = {};
                input.type = INPUT_MOUSE;

                if (isTouching && !currentlyPressed.load()) {
                    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
                    SendInput(1, &input, sizeof(INPUT));
                    currentlyPressed = true;
                }
                else if (!isTouching && currentlyPressed.load()) {
                    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
                    SendInput(1, &input, sizeof(INPUT));
                    currentlyPressed = false;
                }

                lastTouchState = isTouching;
            }
            catch (...) {
            }
        }
    }

    void ParseWacomData(BYTE* data, DWORD length, int& rawX, int& rawY, bool& inProximity, bool& isTouching) {
        if (length < 8 || data == nullptr) return;

        try {
            if (data[0] == 0x01 && length >= 8) {
                rawX = data[2] | (data[3] << 8);
                rawY = data[4] | (data[5] << 8);
                inProximity = (data[1] & 0x01) != 0;
                isTouching = (data[1] & 0x02) != 0;
            }
            else if (data[0] == 0x02 && length >= 8) {
                rawX = data[2] | (data[3] << 8);
                rawY = data[4] | (data[5] << 8);
                inProximity = (data[1] & 0x20) != 0;
                isTouching = (data[1] & 0x01) != 0;
            }
            else if (data[0] != 0x00 && length >= 6) {
                rawX = data[0] | (data[1] << 8);
                rawY = data[2] | (data[3] << 8);
                inProximity = (data[4] & 0x01) != 0;
                isTouching = (data[4] & 0x02) != 0;
            }
            else {
                rawX = rawY = 0;
                inProximity = isTouching = false;
                return;
            }

            if (rawX == 0xFFFF || rawY == 0xFFFF || rawX == 0x7FFF || rawY == 0x7FFF) {
                rawX = rawY = 0;
                inProximity = isTouching = false;
            }
        }
        catch (...) {
            rawX = rawY = 0;
            inProximity = isTouching = false;
        }
    }

    void ParseWacomCTL472Data(BYTE* data, DWORD length, int& rawX, int& rawY, bool& inProximity, bool& isTouching) {
        if (length < 10 || data == nullptr) return;

        try {
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
                rawX = rawY = 0;
                inProximity = isTouching = false;
                return;
            }

            if (rawX == 0xFFFF || rawY == 0xFFFF || rawX == 0x7FFF || rawY == 0x7FFF) {
                rawX = rawY = 0;
                inProximity = isTouching = false;
            }
        }
        catch (...) {
            rawX = rawY = 0;
            inProximity = isTouching = false;
        }
    }

    void ParseXPPenData(BYTE* data, DWORD length, int& rawX, int& rawY, bool& inProximity, bool& isTouching) {
        if (length < 14 || data == nullptr) return;

        try {
            if (data[0] == 0x02 && (data[1] == 0xA0 || data[1] == 0xA1)) {
                rawX = data[2] | (data[3] << 8);
                rawY = data[4] | (data[5] << 8);

                inProximity = true;
                isTouching = (data[1] == 0xA1);
            }
            else {
                rawX = rawY = 0;
                inProximity = isTouching = false;
                return;
            }

            if (rawX == 0xFFFF || rawY == 0xFFFF || rawX == 0x7FFF || rawY == 0x7FFF) {
                rawX = rawY = 0;
                inProximity = isTouching = false;
            }
        }
        catch (...) {
            rawX = rawY = 0;
            inProximity = isTouching = false;
        }
    }

public:
    void WaitForExit() {
        if (configThread.joinable()) {
            configThread.join();
        }
    }
};

HighPerformancePool<TabletData, 128> HighPerformanceTabletDriver::tabletPool;


int main() {
    std::cout << "Ultra Tablet Driver v4.1 (stable) - by vilounos" << std::endl;
    std::cout << "Removed Interpolation, added Smoothing" << std::endl;
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