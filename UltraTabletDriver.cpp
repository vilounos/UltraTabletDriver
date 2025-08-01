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

#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "hid.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "winmm.lib")

struct alignas(16) Vec2f {
    float x, y;
    float _pad[2];

    Vec2f() : x(0.0f), y(0.0f), _pad{ 0.0f, 0.0f } {}
    Vec2f(float x_, float y_) : x(x_), y(y_), _pad{ 0.0f, 0.0f } {}
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

    static Vec2f CubicInterpolate(const Vec2f& p0, const Vec2f& p1, const Vec2f& p2, const Vec2f& p3, float t) {
        __m128 vp0 = _mm_load_ps(reinterpret_cast<const float*>(&p0));
        __m128 vp1 = _mm_load_ps(reinterpret_cast<const float*>(&p1));
        __m128 vp2 = _mm_load_ps(reinterpret_cast<const float*>(&p2));
        __m128 vp3 = _mm_load_ps(reinterpret_cast<const float*>(&p3));

        __m128 vt = _mm_set1_ps(t);
        __m128 vt2 = _mm_mul_ps(vt, vt);
        __m128 vt3 = _mm_mul_ps(vt2, vt);

        __m128 a = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.5f), vp0),
            _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.5f), vp1),
                _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-1.5f), vp2),
                    _mm_mul_ps(_mm_set1_ps(0.5f), vp3))));

        __m128 b = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.0f), vp0),
            _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-2.5f), vp1),
                _mm_add_ps(_mm_mul_ps(_mm_set1_ps(2.0f), vp2),
                    _mm_mul_ps(_mm_set1_ps(-0.5f), vp3))));

        __m128 c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.5f), vp0),
            _mm_mul_ps(_mm_set1_ps(0.5f), vp2));

        __m128 d = vp1;

        __m128 result = _mm_add_ps(_mm_mul_ps(a, vt3),
            _mm_add_ps(_mm_mul_ps(b, vt2),
                _mm_add_ps(_mm_mul_ps(c, vt), d)));

        Vec2f output;
        _mm_store_ps(reinterpret_cast<float*>(&output), result);
        return output;
    }
};

enum class RotationType {
    NONE = 0,
    LEFT = 1,
    FLIP = 2,
    RIGHT = 3
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

struct alignas(16) InterpolationSample {
    Vec2f screenPos;
    std::chrono::high_resolution_clock::time_point timestamp;
    bool valid;
    float _pad;
};

static const int INTERPOLATION_BUFFER_SIZE = 16;
alignas(16) InterpolationSample interpolationBuffer[INTERPOLATION_BUFFER_SIZE];
std::atomic<int> interpolationWriteIndex{ 0 };
std::atomic<int> interpolationReadIndex{ 0 };
std::atomic<int> interpolationSampleCount{ 0 };

bool interpolationEnabled = true;
int interpolationMultiplier = 2;

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
};

struct alignas(16) TabletData {
    Vec2i rawPos;
    bool inProximity;
    bool isTouching;
    bool isValid;
    float _pad;
    std::chrono::high_resolution_clock::time_point timestamp;
};

class HighPerformanceTabletDriver {
private:

    struct alignas(16) AdvancedInterpolationSample {
        Vec2f screenPos;
        Vec2f velocity;
        float pressure;
        std::chrono::high_resolution_clock::time_point timestamp;
        bool valid;
        float _pad[3];
    };

    static const int ADVANCED_INTERPOLATION_BUFFER_SIZE = 32;
    alignas(16) AdvancedInterpolationSample advancedInterpolationBuffer[ADVANCED_INTERPOLATION_BUFFER_SIZE];
    std::atomic<int> advancedInterpolationWriteIndex{ 0 };
    std::atomic<int> advancedInterpolationSampleCount{ 0 };

    std::atomic<float> adaptiveInterpolationRate{ 1.0f };
    std::atomic<float> currentMovementSpeed{ 0.0f };
    std::atomic<bool> useVelocityBasedInterpolation{ true };
    std::atomic<bool> usePredictiveInterpolation{ true };

    enum class InterpolationQuality {
        LINEAR = 0,
        CUBIC = 1,
        BEZIER = 2,
        ADAPTIVE = 3
    };
    InterpolationQuality interpolationQuality = InterpolationQuality::ADAPTIVE;

    bool hasSSE, hasSSE2, hasAVX;

    alignas(16) float rotationCos[4];
    alignas(16) float rotationSin[4];

    std::atomic<double> tabletFrequency{ 0.0 };
    std::atomic<double> programFrequency{ 0.0 };
    std::atomic<double> cursorUpdateFrequency{ 0.0 };
    std::deque<std::chrono::high_resolution_clock::time_point> tabletUpdateTimes;
    std::deque<std::chrono::high_resolution_clock::time_point> programUpdateTimes;
    std::deque<std::chrono::high_resolution_clock::time_point> cursorUpdateTimes;
    std::mutex frequencyMutex;

    enum class LoggingMode {
        OFF = 0,
        FREQUENCY = 1,
        PREDICTION = 2,
        POSITION = 3
    };

    LoggingMode currentLoggingMode = LoggingMode::OFF;
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

    std::atomic<TabletData*> latestData{ nullptr };
    alignas(16) TabletData dataBuffer[2];
    std::atomic<int> currentBuffer{ 0 };

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

public:
    HighPerformanceTabletDriver() : deviceHandle(INVALID_HANDLE_VALUE) {
        hasSSE = SIMDMath::HasSSE();
        hasSSE2 = SIMDMath::HasSSE2();
        hasAVX = SIMDMath::HasAVX();

        std::cout << "SIMD Support: SSE=" << (hasSSE ? "YES" : "NO")
            << ", SSE2=" << (hasSSE2 ? "YES" : "NO")
            << ", AVX=" << (hasAVX ? "YES" : "NO") << std::endl;

        rotationCos[0] = 1.0f; rotationSin[0] = 0.0f;
        rotationCos[1] = 0.0f; rotationSin[1] = 1.0f;
        rotationCos[2] = -1.0f; rotationSin[2] = 0.0f;
        rotationCos[3] = 0.0f; rotationSin[3] = -1.0f;

        currentTablet = TABLET_SPECS[0];
        DetectMonitors();
        LoadConfig();

        consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
        loggingLinePos = { 0, 0 };

        memset(&dataBuffer[0], 0, sizeof(TabletData));
        memset(&dataBuffer[1], 0, sizeof(TabletData));
        dataBuffer[0].isValid = false;
        dataBuffer[1].isValid = false;

        Vec2f zero(0.0f, 0.0f);
        Vec2i zeroI(0, 0);
        currentVelocity.store(zero);
        lastScreenPos.store(zero);
        lastValidRawPos.store(zeroI);
    }

    ~HighPerformanceTabletDriver() {
        Stop();
        if (deviceHandle != INVALID_HANDLE_VALUE) {
            CloseHandle(deviceHandle);
            deviceHandle = INVALID_HANDLE_VALUE;
        }
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
                    else if (key == "interpolationEnabled") interpolationEnabled = (value == "1");
                    else if (key == "interpolationMultiplier") interpolationMultiplier = std::stoi(value);
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
            file << "interpolationEnabled = " << (interpolationEnabled ? 1 : 0) << std::endl;
            file << "interpolationMultiplier = " << interpolationMultiplier << std::endl;
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
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }

    void UpdateTabletFrequency() {
        auto currentTime = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::mutex> lock(frequencyMutex);

        tabletUpdateTimes.push_back(currentTime);
        auto oneSecondAgo = currentTime - std::chrono::seconds(1);
        while (!tabletUpdateTimes.empty() && tabletUpdateTimes.front() < oneSecondAgo) {
            tabletUpdateTimes.pop_front();
        }
        tabletFrequency.store(static_cast<double>(tabletUpdateTimes.size()));
    }

    void UpdateProgramFrequency() {
        auto currentTime = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::mutex> lock(frequencyMutex);

        programUpdateTimes.push_back(currentTime);
        auto oneSecondAgo = currentTime - std::chrono::seconds(1);
        while (!programUpdateTimes.empty() && programUpdateTimes.front() < oneSecondAgo) {
            programUpdateTimes.pop_front();
        }
        programFrequency.store(static_cast<double>(programUpdateTimes.size()));
    }

    void UpdateCursorFrequency() {
        auto currentTime = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::mutex> lock(frequencyMutex);

        cursorUpdateTimes.push_back(currentTime);
        auto oneSecondAgo = currentTime - std::chrono::seconds(1);
        while (!cursorUpdateTimes.empty() && cursorUpdateTimes.front() < oneSecondAgo) {
            cursorUpdateTimes.pop_front();
        }
        cursorUpdateFrequency.store(static_cast<double>(cursorUpdateTimes.size()));
    }

    void UpdateLoggingDisplay() {
        if (currentLoggingMode == LoggingMode::OFF || !running) return;

        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (!GetConsoleScreenBufferInfo(consoleHandle, &csbi)) return;
        COORD savedPos = csbi.dwCursorPosition;

        SetConsoleCursorPosition(consoleHandle, loggingLinePos);

        switch (currentLoggingMode) {
        case LoggingMode::FREQUENCY: {
            double tFreq = tabletFrequency.load();
            double pFreq = programFrequency.load();
            double cFreq = cursorUpdateFrequency.load();
            std::cout << "Tablet: " << (int)tFreq << " Hz | Program: " << (int)pFreq
                << " Hz | Cursor: " << (int)cFreq << " Hz                                           ";
            break;
        }
        case LoggingMode::PREDICTION: {
            Vec2f vel = currentVelocity.load();
            double predAmount = predictionAmount.load();
            double totalVelocity = SIMDMath::Length(vel);
            std::cout << "Velocity: " << (int)totalVelocity << " px/frame | VelX: " << (int)vel.x
                << " | VelY: " << (int)vel.y << " | Prediction: " << (int)predAmount << " px           ";
            break;
        }
        case LoggingMode::POSITION: {
            int tabX = currentTabletX.load();
            int tabY = currentTabletY.load();
            int scrX = currentScreenX.load();
            int scrY = currentScreenY.load();
            bool clicking = isCurrentlyClicking.load();
            std::cout << "Tablet: (" << tabX << "," << tabY << ") | Screen: (" << scrX << "," << scrY
                << ") | Click: " << (clicking ? "YES" : "NO") << "               ";
            break;
        }
        }

        SetConsoleCursorPosition(consoleHandle, savedPos);
        std::cout.flush();
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
    void ShowCurrentSettings() const {
        system("cls");
        std::cout << "=== Ultra Tablet Driver v3.1 (stable) - by vilounos ===" << std::endl;
        std::cout << "Detected Tablet: " << currentTablet.name << std::endl;
        std::cout << "SIMD Acceleration: " << (hasSSE2 ? "ENABLED" : "DISABLED") << std::endl;

        if (monitors.size() > 1) {
            std::cout << "Current Monitor: " << (config.currentMonitor + 1) << "/" << monitors.size()
                << " (" << monitors[config.currentMonitor].name << ") - "
                << monitors[config.currentMonitor].width << "x" << monitors[config.currentMonitor].height << std::endl;
        }
        else {
            std::cout << "Monitor Resolution: " << monitors[0].width << "x" << monitors[0].height << std::endl;
        }

        std::cout << "Tablet Size: " << currentTablet.widthMM << "x" << currentTablet.heightMM << "mm" << std::endl;
        std::cout << "Max Resolution: " << currentTablet.maxX << "x" << currentTablet.maxY << std::endl;
        std::cout << std::endl;

        switch (config.rotation) {
        case 0: std::cout << "Active Area Size: " << config.areaWidth << "x" << config.areaHeight << "mm" << std::endl; break;
        case 1: std::cout << "Active Area Size: " << config.areaHeight << "x" << config.areaWidth << "mm" << std::endl; break;
        case 2: std::cout << "Active Area Size: " << config.areaWidth << "x" << config.areaHeight << "mm" << std::endl; break;
        case 3: std::cout << "Active Area Size: " << config.areaHeight << "x" << config.areaWidth << "mm" << std::endl; break;
        }

        std::cout << "Area Center: (" << config.areaCenterX << "," << config.areaCenterY << ")mm" << std::endl;
        std::cout << "Rotation: ";
        switch (config.rotation) {
        case 0: std::cout << "None (0 degrees)"; break;
        case 1: std::cout << "Left (90 degrees CCW)"; break;
        case 2: std::cout << "Flip (180 degrees)"; break;
        case 3: std::cout << "Right (270 degrees CCW)"; break;
        }
        std::cout << std::endl;
        std::cout << "Movement Prediction: " << (config.movementPrediction ? "ON" : "OFF");
        if (config.movementPrediction) {
            std::cout << " (Strength: " << config.predictionStrength << ")";
        }
        std::cout << std::endl;

        std::cout << "Movement Interpolation: " << (interpolationEnabled ? "ON" : "OFF");
        if (interpolationEnabled) {
            std::cout << " (Multiplier: " << interpolationMultiplier << "x)";
        }
        std::cout << std::endl;

        std::cout << "Click Simulation: " << (config.clickEnabled ? "ON" : "OFF") << std::endl;

        std::cout << "Logging Mode: ";
        switch (currentLoggingMode) {
        case LoggingMode::OFF: std::cout << "OFF"; break;
        case LoggingMode::FREQUENCY: std::cout << "FREQUENCY"; break;
        case LoggingMode::PREDICTION: std::cout << "PREDICTION"; break;
        case LoggingMode::POSITION: std::cout << "POSITION"; break;
        }
        std::cout << std::endl;

        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(consoleHandle, &csbi)) {
            const_cast<HighPerformanceTabletDriver*>(this)->loggingLinePos = csbi.dwCursorPosition;
        }

        std::cout << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "Arrow keys: Move area center" << std::endl;
        std::cout << "2,4,6,8: Adjust area size (width/height)" << std::endl;
        std::cout << "R: Change rotation" << std::endl;
        std::cout << "P: Toggle movement prediction" << std::endl;
        std::cout << "C: Toggle click simulation" << std::endl;
        std::cout << "+/-: Adjust prediction strength" << std::endl;
        std::cout << "I: Toggle interpolation" << std::endl;
        std::cout << "[/]: Adjust interpolation multiplier" << std::endl;
        if (monitors.size() > 1) {
            std::cout << "M: Switch monitor" << std::endl;
        }
        std::cout << "S: Start/Stop driver" << std::endl;
        std::cout << "Q: Quit" << std::endl;
        std::cout << "L: Cycle logging mode (OFF->FREQ->PRED->POS)" << std::endl;
        if (running) {
            std::cout << std::endl << "Driver is running... ESC to restart driver." << std::endl;
        }
    }

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

    void CycleLoggingMode() {
        currentLoggingMode = static_cast<LoggingMode>((static_cast<int>(currentLoggingMode) + 1) % 4);

        if (currentLoggingMode != LoggingMode::OFF) {
            if (running && !loggingThread.joinable()) {
                loggingThread = std::thread(&HighPerformanceTabletDriver::LoggingUpdateLoop, this);
            }
        }
    }

    void TogglePrediction() {
        config.movementPrediction = !config.movementPrediction;
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
        configMode = true;
        ShowCurrentSettings();

        configThread = std::thread([this]() {
            SetThreadSafePriority();

            while (configMode && !emergencyShutdown) {
                if (_kbhit()) {
                    char key = _getch();

                    if (key == -32 || key == 0) {
                        key = _getch();
                        switch (key) {
                        case 72: MoveAreaCenter(0, -1); ShowCurrentSettings(); break;
                        case 80: MoveAreaCenter(0, 1); ShowCurrentSettings(); break;
                        case 75: MoveAreaCenter(-1, 0); ShowCurrentSettings(); break;
                        case 77: MoveAreaCenter(1, 0); ShowCurrentSettings(); break;
                        }
                    }
                    else {
                        switch (key) {
                        case '2': AdjustAreaSize(0, -1); ShowCurrentSettings(); break;
                        case '4': AdjustAreaSize(-1, 0); ShowCurrentSettings(); break;
                        case '6': AdjustAreaSize(1, 0); ShowCurrentSettings(); break;
                        case '8': AdjustAreaSize(0, 1); ShowCurrentSettings(); break;
                        case 'r': case 'R': ConfigureRotation(); ShowCurrentSettings(); break;
                        case 'm': case 'M': SwitchMonitor(); ShowCurrentSettings(); break;
                        case 'p': case 'P': TogglePrediction(); ShowCurrentSettings(); break;
                        case 'c': case 'C': ToggleClick(); ShowCurrentSettings(); break;
                        case '+': case '=': AdjustPredictionStrength(1); ShowCurrentSettings(); break;
                        case '-': case '_': AdjustPredictionStrength(-1); ShowCurrentSettings(); break;
                        case 's': case 'S':
                            if (running) Stop(); else Start();
                            ShowCurrentSettings(); break;
                        case 'q': case 'Q': configMode = false; Stop(); break;
                        case 'l': case 'L': CycleLoggingMode(); ShowCurrentSettings(); break;
                        case 27: if (running) { RestartDriver(); ShowCurrentSettings(); } break;
                        case 'i': case 'I': ToggleInterpolation(); ShowCurrentSettings(); break;
                        case '[': AdjustInterpolationMultiplier(-1); ShowCurrentSettings(); break;
                        case ']': AdjustInterpolationMultiplier(1); ShowCurrentSettings(); break;
                        }
                    }
                }
            }
            });
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

        {
            std::lock_guard<std::mutex> lock(frequencyMutex);
            tabletUpdateTimes.clear();
            programUpdateTimes.clear();
            cursorUpdateTimes.clear();
            tabletFrequency = 0.0;
            programFrequency = 0.0;
            cursorUpdateFrequency = 0.0;
        }

        ClearInterpolationBuffer();

        inputThread = std::thread(&HighPerformanceTabletDriver::SafeInputLoop, this);
        processingThread = std::thread(&HighPerformanceTabletDriver::SafeProcessingLoop, this);

        if (currentLoggingMode != LoggingMode::OFF) {
            loggingThread = std::thread(&HighPerformanceTabletDriver::LoggingUpdateLoop, this);
        }
    }

    void Stop() {
        emergencyShutdown = true;
        running = false;

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
        SetThreadSafePriority();

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
        SetThreadSafePriority();

        TabletData* lastProcessedData = nullptr;

        while (running && !emergencyShutdown) {
            try {
                TabletData* currentData = latestData.load();

                UpdateProgramFrequency();

                if (currentData != nullptr && currentData != lastProcessedData && currentData->isValid) {

                    Vec2f baseScreenPos;
                    if (!CalculateScreenPosition(*currentData, baseScreenPos)) {
                        lastProcessedData = currentData;
                        continue;
                    }

                    Vec2f finalScreenPos;
                    ApplyPrediction(baseScreenPos, finalScreenPos, currentData->timestamp);

                    if (interpolationEnabled && hasLastPosition.load()) {
                        Vec2f currentVel = currentVelocity.load();
                        StoreAdvancedInterpolationSample(finalScreenPos, currentVel, currentData->timestamp);

                        auto interpolatedPositions = GenerateAdvancedInterpolatedPositions();

                        for (const auto& interpPos : interpolatedPositions) {
                            MoveCursorToPosition(interpPos);
                        }
                    }
                    else {
                        if (interpolationEnabled) {
                            Vec2f currentVel = currentVelocity.load();
                            StoreAdvancedInterpolationSample(finalScreenPos, currentVel, currentData->timestamp);
                        }
                        MoveCursorToPosition(finalScreenPos);
                    }

                    HandleMouseClick(currentData->isTouching);

                    lastProcessedData = currentData;
                }
            }
            catch (...) {
                emergencyShutdown = true;
                break;
            }
        }
    }

    bool CalculateScreenPosition(const TabletData& data, Vec2f& screenPos) {
        bool wasJustLifted = wasInProximityLast.load() && !data.inProximity;
        wasInProximityLast.store(data.inProximity);

        if (!data.inProximity && !data.isTouching) {
            hasLastPosition = false;
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

        bool wasJustLifted = wasInProximityLast.load() && !hasLastPosition.load();

        if (config.movementPrediction && hasLastPosition && !wasJustLifted) {
            Vec2f lastPos = lastScreenPos.load();
            Vec2f velocity = SIMDMath::Sub(baseScreenPos, lastPos);

            currentVelocity.store(velocity);

            float predictionFactor = min(config.predictionStrength * 0.15f, 3.0f);
            Vec2f prediction = SIMDMath::Scale(velocity, predictionFactor);

            finalScreenPos = SIMDMath::Add(baseScreenPos, prediction);

            predictionAmount.store(SIMDMath::Length(prediction));
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

            INPUT input = {};
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_MOVE;
            input.mi.dx = deltaX;
            input.mi.dy = deltaY;
            SendInput(1, &input, sizeof(INPUT));

            UpdateCursorFrequency();

            currentScreenX.store(screenX);
            currentScreenY.store(screenY);
        }
        catch (...) {
        }
    }

    void ToggleInterpolation() {
        interpolationEnabled = !interpolationEnabled;

        if (!interpolationEnabled) {
            ClearInterpolationBuffer();
        }

        SaveConfig();
    }

    void AdjustInterpolationMultiplier(int delta) {
        interpolationMultiplier = max(1, min(25, interpolationMultiplier + delta));
        SaveConfig();
    }

    void ClearInterpolationBuffer() {
        interpolationWriteIndex.store(0);
        interpolationReadIndex.store(0);
        interpolationSampleCount.store(0);

        for (int i = 0; i < INTERPOLATION_BUFFER_SIZE; i++) {
            interpolationBuffer[i].valid = false;
        }
    }

    void LoggingUpdateLoop() {
        SetThreadSafePriority();

        while (running && !emergencyShutdown) {
            try {
                UpdateLoggingDisplay();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            catch (...) {
                break;
            }
        }
    }

    void ProcessRawTabletData(BYTE* buffer, DWORD length) {
        if (length < 6 || buffer == nullptr) return;

        int nextBuffer = 1 - currentBuffer.load();
        TabletData* data = &dataBuffer[nextBuffer];

        data->timestamp = std::chrono::high_resolution_clock::now();
        data->inProximity = false;
        data->isTouching = false;
        data->rawPos = Vec2i(0, 0);
        data->isValid = false;

        try {
            int rawX, rawY;
            switch (currentTablet.type) {
            case TabletType::WACOM_CTL672:
                ParseWacomData(buffer, length, rawX, rawY, data->inProximity, data->isTouching);
                break;
            case TabletType::WACOM_CTL472:
                ParseWacomCTL472Data(buffer, length, rawX, rawY, data->inProximity, data->isTouching);
                break;
            case TabletType::XPPEN_STAR_G640:
                ParseXPPenData(buffer, length, rawX, rawY, data->inProximity, data->isTouching);
                break;
            default:
                return;
            }

            data->rawPos = Vec2i(rawX, rawY);
        }
        catch (...) {
            return;
        }

        if (!IsValidTabletData(data->rawPos, data->inProximity)) {
            return;
        }

        data->isValid = true;

        if (data->inProximity || data->isTouching) {
            UpdateTabletFrequency();

            currentBuffer.store(nextBuffer);
            latestData.store(data);
        }
    }

    void StoreAdvancedInterpolationSample(const Vec2f& screenPos, const Vec2f& velocity,
        std::chrono::high_resolution_clock::time_point timestamp) {
        int writeIdx = advancedInterpolationWriteIndex.load();

        advancedInterpolationBuffer[writeIdx].screenPos = screenPos;
        advancedInterpolationBuffer[writeIdx].velocity = velocity;
        advancedInterpolationBuffer[writeIdx].timestamp = timestamp;
        advancedInterpolationBuffer[writeIdx].valid = true;

        advancedInterpolationWriteIndex.store((writeIdx + 1) % ADVANCED_INTERPOLATION_BUFFER_SIZE);

        int currentCount = advancedInterpolationSampleCount.load();
        if (currentCount < ADVANCED_INTERPOLATION_BUFFER_SIZE) {
            advancedInterpolationSampleCount.store(currentCount + 1);
        }
    }

    std::vector<Vec2f> GenerateAdvancedInterpolatedPositions() {
        std::vector<Vec2f> positions;

        int sampleCount = advancedInterpolationSampleCount.load();
        if (sampleCount < 2) {
            return positions;
        }

        std::vector<AdvancedInterpolationSample> recentSamples;
        int writeIdx = advancedInterpolationWriteIndex.load();

        for (int i = min(sampleCount, 8); i > 0; i--) {
            int idx = (writeIdx - i + ADVANCED_INTERPOLATION_BUFFER_SIZE) % ADVANCED_INTERPOLATION_BUFFER_SIZE;
            if (advancedInterpolationBuffer[idx].valid) {
                recentSamples.push_back(advancedInterpolationBuffer[idx]);
            }
        }

        if (recentSamples.size() < 2) return positions;

        float avgSpeed = CalculateAverageSpeed(recentSamples);
        float acceleration = CalculateAcceleration(recentSamples);
        bool isRapidMovement = avgSpeed > 100.0f;
        bool isDecelerating = acceleration < -10.0f;

        currentMovementSpeed.store(avgSpeed);

        int steps = interpolationMultiplier;
        if (useVelocityBasedInterpolation.load()) {
            if (isRapidMovement) {
                steps = min(steps * 2, 8);
            }
            else if (avgSpeed < 20.0f) {
                steps = max(steps / 2, 1);
            }
        }

        const auto& startSample = recentSamples[recentSamples.size() - 2];
        const auto& endSample = recentSamples[recentSamples.size() - 1];

        switch (interpolationQuality) {
        case InterpolationQuality::LINEAR:
            positions = GenerateLinearInterpolation(startSample, endSample, steps);
            break;

        case InterpolationQuality::CUBIC:
            if (recentSamples.size() >= 4) {
                positions = GenerateCubicInterpolation(recentSamples, steps);
            }
            else {
                positions = GenerateLinearInterpolation(startSample, endSample, steps);
            }
            break;

        case InterpolationQuality::BEZIER:
            positions = GenerateBezierInterpolation(startSample, endSample, steps);
            break;

        case InterpolationQuality::ADAPTIVE:
        default:
            if (isRapidMovement && recentSamples.size() >= 4) {
                positions = GenerateCubicInterpolation(recentSamples, steps);
            }
            else if (isDecelerating) {
                positions = GenerateBezierInterpolation(startSample, endSample, steps);
            }
            else {
                positions = GenerateLinearInterpolation(startSample, endSample, steps);
            }
            break;
        }

        if (usePredictiveInterpolation.load() && !positions.empty()) {
            ApplyPredictiveSmoothing(positions, endSample.velocity);
        }

        return positions;
    }

    float CalculateAverageSpeed(const std::vector<AdvancedInterpolationSample>& samples) {
        if (samples.size() < 2) return 0.0f;

        float totalSpeed = 0.0f;
        for (size_t i = 1; i < samples.size(); i++) {
            float speed = SIMDMath::Length(SIMDMath::Sub(samples[i].screenPos, samples[i - 1].screenPos));
            totalSpeed += speed;
        }
        return totalSpeed / (samples.size() - 1);
    }

    float CalculateAcceleration(const std::vector<AdvancedInterpolationSample>& samples) {
        if (samples.size() < 3) return 0.0f;

        Vec2f vel1 = SIMDMath::Sub(samples[samples.size() - 2].screenPos, samples[samples.size() - 3].screenPos);
        Vec2f vel2 = SIMDMath::Sub(samples[samples.size() - 1].screenPos, samples[samples.size() - 2].screenPos);

        float speed1 = SIMDMath::Length(vel1);
        float speed2 = SIMDMath::Length(vel2);

        return speed2 - speed1;
    }

    std::vector<Vec2f> GenerateLinearInterpolation(const AdvancedInterpolationSample& start,
        const AdvancedInterpolationSample& end, int steps) {
        std::vector<Vec2f> positions;

        for (int i = 1; i <= steps; i++) {
            float t = (float)i / (steps + 1);
            Vec2f interpPos = SIMDMath::Lerp(start.screenPos, end.screenPos, t);
            positions.push_back(interpPos);
        }

        return positions;
    }

    std::vector<Vec2f> GenerateCubicInterpolation(const std::vector<AdvancedInterpolationSample>& samples, int steps) {
        std::vector<Vec2f> positions;

        if (samples.size() < 4) return positions;

        const auto& p0 = samples[samples.size() - 4].screenPos;
        const auto& p1 = samples[samples.size() - 3].screenPos;
        const auto& p2 = samples[samples.size() - 2].screenPos;
        const auto& p3 = samples[samples.size() - 1].screenPos;

        for (int i = 1; i <= steps; i++) {
            float t = (float)i / (steps + 1);
            Vec2f interpPos = SIMDMath::CubicInterpolate(p0, p1, p2, p3, t);
            positions.push_back(interpPos);
        }

        return positions;
    }

    std::vector<Vec2f> GenerateBezierInterpolation(const AdvancedInterpolationSample& start,
        const AdvancedInterpolationSample& end, int steps) {
        std::vector<Vec2f> positions;

        Vec2f controlPoint1 = SIMDMath::Add(start.screenPos, SIMDMath::Scale(start.velocity, 0.3f));
        Vec2f controlPoint2 = SIMDMath::Sub(end.screenPos, SIMDMath::Scale(end.velocity, 0.3f));

        for (int i = 1; i <= steps; i++) {
            float t = (float)i / (steps + 1);
            Vec2f interpPos = CubicBezier(start.screenPos, controlPoint1, controlPoint2, end.screenPos, t);
            positions.push_back(interpPos);
        }

        return positions;
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

    void ApplyPredictiveSmoothing(std::vector<Vec2f>& positions, const Vec2f& velocity) {
        if (positions.empty()) return;

        float velocityMagnitude = SIMDMath::Length(velocity);
        if (velocityMagnitude < 3.0f) return;

        for (size_t i = positions.size() / 2; i < positions.size(); i++) {
            float predictionFactor = 0.1f * (float)(i - positions.size() / 2) / (positions.size() / 2);
            Vec2f prediction = SIMDMath::Scale(velocity, predictionFactor);
            positions[i] = SIMDMath::Add(positions[i], prediction);
        }
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

        Vec2i halfSize(
            (config.areaWidth * currentTablet.maxX / currentTablet.widthMM) / 2,
            (config.areaHeight * currentTablet.maxY / currentTablet.heightMM) / 2
        );

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

int main() {
    std::cout << "Ultra Tablet Driver v3.1 (stable) - by vilounos" << std::endl;
    std::cout << "Added SIMD Acceleration" << std::endl;
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