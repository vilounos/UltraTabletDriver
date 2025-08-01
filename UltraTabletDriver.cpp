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

#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "hid.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "winmm.lib")

enum class RotationType {
    NONE = 0, // No rotation
    LEFT = 1, // 90° counter-clockwise
    FLIP = 2, // 180°
    RIGHT = 3 // 270° counter-clockwise (90° clockwise)
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

struct TabletData {
    int rawX;
    int rawY;
    bool inProximity;
    bool isTouching;
    bool isValid;
    std::chrono::high_resolution_clock::time_point timestamp;
};

class HighPerformanceTabletDriver {
private:
    // High-performance timing variables
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

    // Prediction logging variables
    std::atomic<double> currentVelocityX{ 0.0 };
    std::atomic<double> currentVelocityY{ 0.0 };
    std::atomic<double> predictionAmount{ 0.0 };

    // Position logging variables
    std::atomic<int> currentTabletX{ 0 };
    std::atomic<int> currentTabletY{ 0 };
    std::atomic<int> currentScreenX{ 0 };
    std::atomic<int> currentScreenY{ 0 };
    std::atomic<bool> isCurrentlyClicking{ false };

    // Threading and performance
    HANDLE deviceHandle;
    std::atomic<bool> running{ false };
    std::atomic<bool> configMode{ false };
    TabletSpec currentTablet;
    DriverConfig config;

    // Safe priority threads
    std::thread inputThread;
    std::thread processingThread;
    std::thread configThread;
    std::thread loggingThread;

    // Lock-free data exchange
    std::atomic<TabletData*> latestData{ nullptr };
    TabletData dataBuffer[2];  // Double buffering
    std::atomic<int> currentBuffer{ 0 };

    // Multi-monitor support
    std::vector<Monitor> monitors;
    int SCREEN_WIDTH = GetSystemMetrics(SM_CXSCREEN);
    int SCREEN_HEIGHT = GetSystemMetrics(SM_CYSCREEN);

    // Movement prediction variables
    std::atomic<int> lastScreenX{ 0 }, lastScreenY{ 0 };
    std::atomic<bool> hasLastPosition{ false };
    std::atomic<bool> wasInProximityLast{ false };

    // Click state tracking
    std::atomic<bool> lastTouchState{ false };
    std::atomic<bool> currentlyPressed{ false };

    // Safety shutdown flag
    std::atomic<bool> emergencyShutdown{ false };

    std::atomic<int> lastValidRawX{ -1 };
    std::atomic<int> lastValidRawY{ -1 };
    std::atomic<bool> hasValidData{ false };

    // Tablet specifications
    const TabletSpec TABLET_SPECS[4] = {
        {0, 0, 0, 0, TabletType::UNKNOWN, "Unknown"},
        {21610, 13498, 216, 135, TabletType::WACOM_CTL672, "Wacom CTL-672"},
        {30480, 20320, 152, 102, TabletType::XPPEN_STAR_G640, "XPPen Star G 640"},
        {15200, 9500, 152, 95, TabletType::WACOM_CTL472, "Wacom CTL-472"}
    };

public:
    HighPerformanceTabletDriver() : deviceHandle(INVALID_HANDLE_VALUE) {
        currentTablet = TABLET_SPECS[0]; // Unknown by default
        DetectMonitors();
        LoadConfig();

        // Get console handle for logging display
        consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
        loggingLinePos = { 0, 0 };

        // Initialize data buffers
        dataBuffer[0] = {};
        dataBuffer[1] = {};
        dataBuffer[0].isValid = false;
        dataBuffer[1].isValid = false;
    }

    ~HighPerformanceTabletDriver() {
        Stop();
        // Ensure all threads are properly cleaned up
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

        // Sort monitors: primary first, then by position
        std::sort(monitors.begin(), monitors.end(), [](const Monitor& a, const Monitor& b) {
            if (a.isPrimary != b.isPrimary) return a.isPrimary;
            return a.x < b.x;
            });

        if (monitors.empty()) {
            // Fallback to single monitor
            Monitor fallback;
            fallback.x = 0;
            fallback.y = 0;
            fallback.width = SCREEN_WIDTH;
            fallback.height = SCREEN_HEIGHT;
            fallback.name = "Primary";
            fallback.isPrimary = true;
            monitors.push_back(fallback);
        }

        // Ensure current monitor is valid
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
                continue; // Skip if memory allocation failed
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

                            // Set default area based on tablet type if config is default
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

        // Save current cursor position
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (!GetConsoleScreenBufferInfo(consoleHandle, &csbi)) return;
        COORD savedPos = csbi.dwCursorPosition;

        // Update display based on current mode
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
            double velX = currentVelocityX.load();
            double velY = currentVelocityY.load();
            double predAmount = predictionAmount.load();
            double totalVelocity = sqrt(velX * velX + velY * velY);
            std::cout << "Velocity: " << (int)totalVelocity << " px/frame | VelX: " << (int)velX
                << " | VelY: " << (int)velY << " | Prediction: " << (int)predAmount << " px           ";
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

        // Restore cursor position
        SetConsoleCursorPosition(consoleHandle, savedPos);
        std::cout.flush();
    }

    TabletType DetectTabletType(USHORT vendorId, USHORT productId) {
        // Wacom devices
        if (vendorId == 0x056A) {
            return PromptWacomTabletSelection();
        }
        // XPPen devices
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
        std::cout << "=== Ultra Tablet Driver v2.1 - by vilounos (stable) ===" << std::endl;
        std::cout << "Detected Tablet: " << currentTablet.name << std::endl;

        // Show monitor information
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

        // When rotated 90 or 270 degrees, swap width and height
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

        std::cout << "Click Simulation: " << (config.clickEnabled ? "ON" : "OFF") << std::endl;

        std::cout << "Logging Mode: ";
        switch (currentLoggingMode) {
        case LoggingMode::OFF: std::cout << "OFF"; break;
        case LoggingMode::FREQUENCY: std::cout << "FREQUENCY"; break;
        case LoggingMode::PREDICTION: std::cout << "PREDICTION"; break;
        case LoggingMode::POSITION: std::cout << "POSITION"; break;
        }
        std::cout << std::endl;

        // Set logging line position
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(consoleHandle, &csbi)) {
            const_cast<HighPerformanceTabletDriver*>(this)->loggingLinePos = csbi.dwCursorPosition;
        }

        if (currentLoggingMode != LoggingMode::OFF) {
            switch (currentLoggingMode) {
            case LoggingMode::FREQUENCY:
                std::cout << "Tablet: 0 Hz | Program: 0 Hz | Cursor: 0 Hz" << std::endl;
                break;
            case LoggingMode::PREDICTION:
                std::cout << "Velocity: 0 px/frame | VelX: 0 | VelY: 0 | Prediction: 0 px" << std::endl;
                break;
            case LoggingMode::POSITION:
                std::cout << "Tablet: (0,0) | Screen: (0,0) | Click: NO" << std::endl;
                break;
            }
        }

        std::cout << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "Arrow keys: Move area center" << std::endl;
        std::cout << "2,4,6,8: Adjust area size (width/height)" << std::endl;
        std::cout << "R: Change rotation" << std::endl;
        std::cout << "P: Toggle movement prediction" << std::endl;
        std::cout << "C: Toggle click simulation" << std::endl;
        std::cout << "+/-: Adjust prediction strength" << std::endl;
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

        // Clamp to tablet bounds
        config.areaCenterX = max(config.areaWidth / 2, min(currentTablet.widthMM - config.areaWidth / 2, config.areaCenterX));
        config.areaCenterY = max(config.areaHeight / 2, min(currentTablet.heightMM - config.areaHeight / 2, config.areaCenterY));
        SaveConfig();
    }

    void AdjustAreaSize(int widthDelta, int heightDelta) {
        config.areaWidth = max(5, min(currentTablet.widthMM, config.areaWidth + widthDelta));
        config.areaHeight = max(5, min(currentTablet.heightMM, config.areaHeight + heightDelta));

        // Adjust center if area goes out of bounds
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
                        case 72:
                            MoveAreaCenter(0, -1);
                            ShowCurrentSettings();
                            break;
                        case 80:
                            MoveAreaCenter(0, 1);
                            ShowCurrentSettings();
                            break;
                        case 75:
                            MoveAreaCenter(-1, 0);
                            ShowCurrentSettings();
                            break;
                        case 77:
                            MoveAreaCenter(1, 0);
                            ShowCurrentSettings();
                            break;
                        }
                    }
                    else {
                        switch (key) {
                        case '2': // Decrease height
                            AdjustAreaSize(0, -1);
                            ShowCurrentSettings();
                            break;
                        case '4': // Decrease width
                            AdjustAreaSize(-1, 0);
                            ShowCurrentSettings();
                            break;
                        case '6': // Increase width
                            AdjustAreaSize(1, 0);
                            ShowCurrentSettings();
                            break;
                        case '8': // Increase height
                            AdjustAreaSize(0, 1);
                            ShowCurrentSettings();
                            break;
                        case 'r':
                        case 'R': // Rotate
                            ConfigureRotation();
                            ShowCurrentSettings();
                            break;
                        case 'm':
                        case 'M': // Switch monitor
                            SwitchMonitor();
                            ShowCurrentSettings();
                            break;
                        case 'p':
                        case 'P': // Toggle prediction
                            TogglePrediction();
                            ShowCurrentSettings();
                            break;
                        case 'c':
                        case 'C': // Toggle click
                            ToggleClick();
                            ShowCurrentSettings();
                            break;
                        case '+':
                        case '=': // Increase prediction
                            AdjustPredictionStrength(1);
                            ShowCurrentSettings();
                            break;
                        case '-':
                        case '_': // Decrease prediction
                            AdjustPredictionStrength(-1);
                            ShowCurrentSettings();
                            break;
                        case 's':
                        case 'S': // Start/Stop
                            if (running) {
                                Stop();
                            }
                            else {
                                Start();
                            }
                            ShowCurrentSettings();
                            break;
                        case 'q':
                        case 'Q': // Quit
                            configMode = false;
                            Stop();
                            break;
                        case 'l':
                        case 'L': // Cycle logging mode
                            CycleLoggingMode();
                            ShowCurrentSettings();
                            break;
                        case 27: // ESC key
                            if (running) {
                                RestartDriver();
                                ShowCurrentSettings();
                            }
                            break;
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
        lastValidRawX = -1;
        lastValidRawY = -1;

        // Clear frequency data
        {
            std::lock_guard<std::mutex> lock(frequencyMutex);
            tabletUpdateTimes.clear();
            programUpdateTimes.clear();
            cursorUpdateTimes.clear();
            tabletFrequency = 0.0;
            programFrequency = 0.0;
            cursorUpdateFrequency = 0.0;
        }

        // Start safe priority threads
        inputThread = std::thread(&HighPerformanceTabletDriver::SafeInputLoop, this);
        processingThread = std::thread(&HighPerformanceTabletDriver::SafeProcessingLoop, this);

        if (currentLoggingMode != LoggingMode::OFF) {
            loggingThread = std::thread(&HighPerformanceTabletDriver::LoggingUpdateLoop, this);
        }
    }

    void Stop() {
        emergencyShutdown = true;
        running = false;

        // Release any pressed mouse button before stopping
        if (currentlyPressed && config.clickEnabled) {
            INPUT input = {};
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
            currentlyPressed = false;
        }

        // Wait for threads to finish safely
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
    // SAFE INPUT LOOP
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
                // Emergency shutdown on any exception
                emergencyShutdown = true;
                break;
            }
        }
    }

    // SAFE PROCESSING LOOP
    void SafeProcessingLoop() {
        SetThreadSafePriority();

        TabletData* lastProcessedData = nullptr;
        auto lastUpdate = std::chrono::high_resolution_clock::now();

        while (running && !emergencyShutdown) {
            try {
                TabletData* currentData = latestData.load();

                UpdateProgramFrequency();

                if (currentData != nullptr && currentData != lastProcessedData && currentData->isValid) {
                    ProcessTabletMovement(*currentData);
                    lastProcessedData = currentData;
                }
            }
            catch (...) {
                // Emergency shutdown on any exception
                emergencyShutdown = true;
                break;
            }
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

        // Get next buffer
        int nextBuffer = 1 - currentBuffer.load();
        TabletData* data = &dataBuffer[nextBuffer];

        data->timestamp = std::chrono::high_resolution_clock::now();
        data->inProximity = false;
        data->isTouching = false;
        data->rawX = 0;
        data->rawY = 0;
        data->isValid = false;

        // Parse data based on tablet type
        try {
            switch (currentTablet.type) {
            case TabletType::WACOM_CTL672:
                ParseWacomData(buffer, length, data->rawX, data->rawY, data->inProximity, data->isTouching);
                break;
            case TabletType::WACOM_CTL472:
                ParseWacomCTL472Data(buffer, length, data->rawX, data->rawY, data->inProximity, data->isTouching);
                break;
            case TabletType::XPPEN_STAR_G640:
                ParseXPPenData(buffer, length, data->rawX, data->rawY, data->inProximity, data->isTouching);
                break;
            default:
                return;
            }
        }
        catch (...) {
            // Skip this data packet on parsing error
            return;
        }

        if (!IsValidTabletData(data->rawX, data->rawY, data->inProximity)) {
            return;
        }

        data->isValid = true;

        if (data->inProximity || data->isTouching) {
            UpdateTabletFrequency();

            // Atomically swap buffers
            currentBuffer.store(nextBuffer);
            latestData.store(data);
        }
    }

    bool IsValidTabletData(int rawX, int rawY, bool inProximity) {
        if (rawX < 0 || rawY < 0 || rawX > currentTablet.maxX || rawY > currentTablet.maxY) {
            return false;
        }

        if ((rawX == 0 && rawY == 0) && !inProximity) {
            return false;
        }

        if (rawX == 0xFFFF || rawY == 0xFFFF || rawX == 0x7FFF || rawY == 0x7FFF) {
            return false;
        }

        if (hasValidData.load()) {
            int lastX = lastValidRawX.load();
            int lastY = lastValidRawY.load();
            
            if (lastX >= 0 && lastY >= 0) {
                int deltaX = abs(rawX - lastX);
                int deltaY = abs(rawY - lastY);

                int maxJumpX = (currentTablet.maxX * 80) / 100;
                int maxJumpY = (currentTablet.maxY * 80) / 100;

                if (deltaX > maxJumpX || deltaY > maxJumpY) {
                    return false;
                }
            }
        }

        lastValidRawX.store(rawX);
        lastValidRawY.store(rawY);
        hasValidData.store(true);

        return true;
    }

    void ProcessTabletMovement(const TabletData& data) {
        if (!data.isValid) {
            return;
        }

        // Check if pen was just lifted
        bool wasJustLifted = wasInProximityLast.load() && !data.inProximity;
        wasInProximityLast.store(data.inProximity);

        if (!data.inProximity && !data.isTouching) {
            hasLastPosition = false;
            return;
        }

        // Calculate current area bounds
        int areaLeft, areaRight, areaTop, areaBottom;
        CalculateAreaBounds(areaLeft, areaRight, areaTop, areaBottom);

        // Get current monitor
        const Monitor& currentMonitor = monitors[config.currentMonitor];

        // Store tablet position for logging
        currentTabletX.store(data.rawX);
        currentTabletY.store(data.rawY);
        isCurrentlyClicking.store(data.isTouching);

        // Check if point is within mapping area
        if (data.rawX < areaLeft || data.rawX > areaRight || data.rawY < areaTop || data.rawY > areaBottom) {
            MoveToNearestEdge(data.rawX, data.rawY, areaLeft, areaRight, areaTop, areaBottom, currentMonitor);

            if (config.clickEnabled && currentlyPressed) {
                HandleMouseClick(false);
            }
            hasLastPosition = false;
            return;
        }

        HandleMouseClick(data.isTouching);

        // Normalize coordinates
        double normalizedX = (double)(data.rawX - areaLeft) / (areaRight - areaLeft);
        double normalizedY = (double)(data.rawY - areaTop) / (areaBottom - areaTop);

        // Apply rotation
        double rotatedNormX, rotatedNormY;
        ApplyRotation(normalizedX, normalizedY, rotatedNormX, rotatedNormY);

        // Map to screen coordinates
        int baseScreenX = currentMonitor.x + (int)(rotatedNormX * currentMonitor.width);
        int baseScreenY = currentMonitor.y + (int)(rotatedNormY * currentMonitor.height);

        // Calculate final position with prediction
        int finalScreenX = baseScreenX;
        int finalScreenY = baseScreenY;

        if (config.movementPrediction && hasLastPosition && !wasJustLifted) {
            // Calculate movement vector
            double velocityX = baseScreenX - lastScreenX.load();
            double velocityY = baseScreenY - lastScreenY.load();

            // Store velocity for logging
            currentVelocityX.store(velocityX);
            currentVelocityY.store(velocityY);

            // Apply prediction with safer limits
            double predictionFactor = min(config.predictionStrength * 0.15, 3.0);
            double predX = velocityX * predictionFactor;
            double predY = velocityY * predictionFactor;

            finalScreenX += (int)predX;
            finalScreenY += (int)predY;

            // Store prediction amount for logging
            predictionAmount.store(sqrt(predX * predX + predY * predY));
        }
        else {
            // Reset prediction logging when not predicting
            currentVelocityX.store(0.0);
            currentVelocityY.store(0.0);
            predictionAmount.store(0.0);
        }

        // Clamp to current monitor bounds with safety margin
        int safeMargin = 5;
        finalScreenX = max(currentMonitor.x + safeMargin,
            min(currentMonitor.x + currentMonitor.width - safeMargin, finalScreenX));
        finalScreenY = max(currentMonitor.y + safeMargin,
            min(currentMonitor.y + currentMonitor.height - safeMargin, finalScreenY));

        if ((finalScreenX <= currentMonitor.x + 10 && finalScreenY <= currentMonitor.y + 10) ||
            (finalScreenX >= currentMonitor.x + currentMonitor.width - 10 &&
                finalScreenY <= currentMonitor.y + 10)) {
            return;
        }

        // Use safer cursor positioning
        try {
            SetCursorPos(finalScreenX, finalScreenY);
            UpdateCursorFrequency(); // Track cursor update frequency

            // Store screen position for logging
            currentScreenX.store(finalScreenX);
            currentScreenY.store(finalScreenY);
        }
        catch (...) {
            // Ignore cursor positioning errors
        }

        // Store current unpredicted screen position
        lastScreenX.store(baseScreenX);
        lastScreenY.store(baseScreenY);
        hasLastPosition = true;
    }

    void MoveToNearestEdge(int rawX, int rawY, int areaLeft, int areaRight, int areaTop, int areaBottom, const Monitor& currentMonitor) {
        // Find the nearest point on the area boundary
        int nearestX = max(areaLeft, min(areaRight, rawX));
        int nearestY = max(areaTop, min(areaBottom, rawY));

        // Normalize the nearest point
        double normalizedX = (double)(nearestX - areaLeft) / (areaRight - areaLeft);
        double normalizedY = (double)(nearestY - areaTop) / (areaBottom - areaTop);

        // Apply rotation
        double rotatedNormX, rotatedNormY;
        ApplyRotation(normalizedX, normalizedY, rotatedNormX, rotatedNormY);

        // Map to screen coordinates
        int screenX = currentMonitor.x + (int)(rotatedNormX * currentMonitor.width);
        int screenY = currentMonitor.y + (int)(rotatedNormY * currentMonitor.height);

        // Clamp to monitor bounds
        screenX = max(currentMonitor.x, min(currentMonitor.x + currentMonitor.width - 1, screenX));
        screenY = max(currentMonitor.y, min(currentMonitor.y + currentMonitor.height - 1, screenY));

        try {
            SetCursorPos(screenX, screenY);
        }
        catch (...) {
            // Ignore cursor positioning errors
        }
    }

    void CalculateAreaBounds(int& left, int& right, int& top, int& bottom) const {
        left = (config.areaCenterX - config.areaWidth / 2) * currentTablet.maxX / currentTablet.widthMM;
        right = (config.areaCenterX + config.areaWidth / 2) * currentTablet.maxX / currentTablet.widthMM;
        top = (config.areaCenterY - config.areaHeight / 2) * currentTablet.maxY / currentTablet.heightMM;
        bottom = (config.areaCenterY + config.areaHeight / 2) * currentTablet.maxY / currentTablet.heightMM;
    }

    void ApplyRotation(double normalizedX, double normalizedY, double& rotatedX, double& rotatedY) const {
        switch (config.rotation) {
        case 0: // NONE
            rotatedX = normalizedX;
            rotatedY = normalizedY;
            break;
        case 1: // LEFT
            rotatedX = 1.0 - normalizedY;
            rotatedY = normalizedX;
            break;
        case 2: // FLIP
            rotatedX = 1.0 - normalizedX;
            rotatedY = 1.0 - normalizedY;
            break;
        case 3: // RIGHT
            rotatedX = normalizedY;
            rotatedY = 1.0 - normalizedX;
            break;
        }
    }

    void HandleMouseClick(bool isTouching) {
        if (!config.clickEnabled) return;

        // Check for state change
        bool lastTouch = lastTouchState.load();
        if (isTouching != lastTouch) {
            try {
                INPUT input = {};
                input.type = INPUT_MOUSE;

                if (isTouching && !currentlyPressed.load()) {
                    // Press down
                    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
                    SendInput(1, &input, sizeof(INPUT));
                    currentlyPressed = true;
                }
                else if (!isTouching && currentlyPressed.load()) {
                    // Release
                    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
                    SendInput(1, &input, sizeof(INPUT));
                    currentlyPressed = false;
                }

                lastTouchState = isTouching;
            }
            catch (...) {
                // Ignore mouse input errors
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
    std::cout << "Ultra Tablet Driver v2.1 - by vilounos (stable)" << std::endl;
    std::cout << "Optimized for stability and safe high performance" << std::endl;
    std::cout << "Support for Wacom CTL-672, CTL-472 & XPPen Star G640" << std::endl << std::endl;

    timeBeginPeriod(1);

    HighPerformanceTabletDriver driver;

    if (!driver.Initialize()) {
        std::cerr << "Failed to initialize tablet driver" << std::endl;
        std::cout << "Make sure your supported tablet is connected:" << std::endl;
        std::cout << "- Wacom CTL-672 (One by Wacom Medium)" << std::endl;
        std::cout << "- Wacom CTL-472 (One by Wacom Small)" << std::endl;
        std::cout << "- XPPen Star G 640" << std::endl;
        std::cout << "If your tablet is connected and all drivers are disabled:" << std::endl;
        std::cout << "- Install and run OpenTabletDriver and than close it (make sure to close it in system tray too)" << std::endl;
        std::cout << "- Make sure you are running this app ad Administrator" << std::endl;
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