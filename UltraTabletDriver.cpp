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

#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "hid.lib")
#pragma comment(lib, "user32.lib")

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

struct DriverConfig {
    int areaWidth = 28;
    int areaHeight = 22;
    int areaCenterX = 80;
    int areaCenterY = 80;
    int rotation = 0;
    bool movementPrediction = false;
    int predictionStrength = 2;
    bool clickEnabled = false;
};

class UniversalTabletDriver {
private:
    HANDLE deviceHandle;
    bool running;
    bool configMode;
    TabletSpec currentTablet;
    DriverConfig config;
    std::thread inputThread;
    std::thread configThread;

    // Screen resolution
    int SCREEN_WIDTH = GetSystemMetrics(SM_CXSCREEN);
    int SCREEN_HEIGHT = GetSystemMetrics(SM_CYSCREEN);

    // Movement prediction variables
    int lastScreenX, lastScreenY;
    bool hasLastPosition;

    // Click state tracking
    bool lastTouchState;
    bool currentlyPressed;

    // Tablet specifications
    const TabletSpec TABLET_SPECS[4] = {
        {0, 0, 0, 0, TabletType::UNKNOWN, "Unknown"},
        {21610, 13498, 216, 135, TabletType::WACOM_CTL672, "Wacom CTL-672"},
        {30480, 20320, 152, 102, TabletType::XPPEN_STAR_G640, "XPPen Star G 640"},
        {15200, 9500, 152, 95, TabletType::WACOM_CTL472, "Wacom CTL-472"}
    };

public:
    UniversalTabletDriver() : deviceHandle(INVALID_HANDLE_VALUE), running(false), configMode(false),
        lastScreenX(0), lastScreenY(0), hasLastPosition(false), lastTouchState(false), currentlyPressed(false) {
        currentTablet = TABLET_SPECS[0]; // Unknown by default
        LoadConfig();
    }

    ~UniversalTabletDriver() {
        Stop();
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
            deviceInterfaceDetailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

            if (SetupDiGetDeviceInterfaceDetail(deviceInfoSet, &deviceInterfaceData,
                deviceInterfaceDetailData, requiredSize,
                NULL, NULL)) {

                HANDLE testHandle = CreateFile(deviceInterfaceDetailData->DevicePath,
                    GENERIC_READ | GENERIC_WRITE,
                    0 | 0,
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
        std::cout << "=== Ultra Tablet Driver v1.0 (stable) - by vilounos ===" << std::endl;
        std::cout << "Detected Tablet: " << currentTablet.name << std::endl;
        std::cout << "Monitor Resolution: " << SCREEN_WIDTH << "x" << SCREEN_HEIGHT << std::endl;
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
        std::cout << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "Arrow keys: Move area center" << std::endl;
        std::cout << "2,4,6,8: Adjust area size (width/height)" << std::endl;
        std::cout << "R: Change rotation" << std::endl;
        std::cout << "P: Toggle movement prediction" << std::endl;
        std::cout << "C: Toggle click simulation" << std::endl;
        std::cout << "+/-: Adjust prediction strength" << std::endl;
        std::cout << "S: Start/Stop driver" << std::endl;
        std::cout << "Q: Quit" << std::endl;
        if (running) {
            std::cout << std::endl << "Driver is RUNNING. ESC to restart driver." << std::endl;
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
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        Start();
    }

    void RunConfiguration() {
        configMode = true;
        ShowCurrentSettings();

        configThread = std::thread([this]() {
            while (configMode) {
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
                        case 27: // ESC key
                            if (running) {
                                RestartDriver();
                                ShowCurrentSettings();
                            }
                            break;
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            });
    }

    void Start() {
        if (deviceHandle == INVALID_HANDLE_VALUE) {
            std::cerr << "Device not initialized" << std::endl;
            return;
        }

        running = true;
        inputThread = std::thread(&UniversalTabletDriver::ProcessInput, this);
        hasLastPosition = false;
        lastTouchState = false;
        currentlyPressed = false;
    }

    void Stop() {
        running = false;
        // Release any pressed mouse button before stopping
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
    }

private:
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
        if (isTouching != lastTouchState) {
            INPUT input = {};
            input.type = INPUT_MOUSE;

            if (isTouching && !currentlyPressed) {
                // Press down
                input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
                SendInput(1, &input, sizeof(INPUT));
                currentlyPressed = true;
            }
            else if (!isTouching && currentlyPressed) {
                // Release
                input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
                SendInput(1, &input, sizeof(INPUT));
                currentlyPressed = false;
            }

            lastTouchState = isTouching;
        }
    }

    void ProcessInput() {
        BYTE buffer[64];
        DWORD bytesRead;

        while (running) {
            if (ReadFile(deviceHandle, buffer, sizeof(buffer), &bytesRead, NULL)) {
                ProcessTabletData(buffer, bytesRead);
            }
            else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    void ProcessTabletData(BYTE* data, DWORD length) {
        if (length < 6) return;

        int rawX = 0, rawY = 0;
        bool inProximity = false;
        bool isTouching = false;

        // Parse data based on tablet type
        switch (currentTablet.type) {
        case TabletType::WACOM_CTL672:
            ParseWacomData(data, length, rawX, rawY, inProximity, isTouching);
            break;
        case TabletType::WACOM_CTL472:
            ParseWacomCTL472Data(data, length, rawX, rawY, inProximity, isTouching);
            break;
        case TabletType::XPPEN_STAR_G640:
            ParseXPPenData(data, length, rawX, rawY, inProximity, isTouching);
            break;
        default:
            return;
        }

        if (!inProximity && (rawX == 0 && rawY == 0)) return;

        // Calculate current area bounds
        int areaLeft, areaRight, areaTop, areaBottom;
        CalculateAreaBounds(areaLeft, areaRight, areaTop, areaBottom);

        // Check if point is within mapping area
        if (rawX < areaLeft || rawX > areaRight || rawY < areaTop || rawY > areaBottom) {
            if (config.clickEnabled && currentlyPressed) {
                HandleMouseClick(false);
            }
            return;
        }

        // Handle mouse clicking based on touch state
        HandleMouseClick(isTouching);

        // Normalize coordinates
        double normalizedX = (double)(rawX - areaLeft) / (areaRight - areaLeft);
        double normalizedY = (double)(rawY - areaTop) / (areaBottom - areaTop);

        // Apply rotation
        double rotatedNormX, rotatedNormY;
        ApplyRotation(normalizedX, normalizedY, rotatedNormX, rotatedNormY);

        // Map to screen coordinates
        int baseScreenX = (int)(rotatedNormX * SCREEN_WIDTH);
        int baseScreenY = (int)(rotatedNormY * SCREEN_HEIGHT);

        // Calculate final position with prediction
        int finalScreenX = baseScreenX;
        int finalScreenY = baseScreenY;

        if (config.movementPrediction && hasLastPosition) {
            // Calculate movement vector
            double velocityX = baseScreenX - lastScreenX;
            double velocityY = baseScreenY - lastScreenY;

            // Apply prediction
            double predictionFactor = config.predictionStrength * 0.15;
            finalScreenX += (int)(velocityX * predictionFactor);
            finalScreenY += (int)(velocityY * predictionFactor);
        }

        // Store current unpredicted screen position
        lastScreenX = baseScreenX;
        lastScreenY = baseScreenY;
        hasLastPosition = true;

        // Clamp to screen bounds
        finalScreenX = max(0, min(SCREEN_WIDTH - 1, finalScreenX));
        finalScreenY = max(0, min(SCREEN_HEIGHT - 1, finalScreenY));

        SetCursorPos(finalScreenX, finalScreenY);
    }

    void ParseWacomData(BYTE* data, DWORD length, int& rawX, int& rawY, bool& inProximity, bool& isTouching) {
        if (length < 8) return;

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
    }

    void ParseWacomCTL472Data(BYTE* data, DWORD length, int& rawX, int& rawY, bool& inProximity, bool& isTouching) {
        if (length < 10) return;

        if (data[0] == 0x01 || data[0] == 0x02) {
            // Extract X coordinate
            rawX = data[2] | (data[3] << 8);
            // Extract Y coordinate
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
            inProximity = false;
            isTouching = false;
        }
    }

    void ParseXPPenData(BYTE* data, DWORD length, int& rawX, int& rawY, bool& inProximity, bool& isTouching) {
        if (length < 14) return;

        if (data[0] == 0x02 && (data[1] == 0xA0 || data[1] == 0xA1)) {
            // X coordinate
            rawX = data[2] | (data[3] << 8);
            // Y coordinate
            rawY = data[4] | (data[5] << 8);

            // Proximity detection
            inProximity = true;

            // Touch detection
            isTouching = (data[1] == 0xA1);
        }
        else {
            inProximity = false;
            isTouching = false;
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
    std::cout << "Ultra Tablet Driver v1.0 (stable) - by vilounos" << std::endl;
    std::cout << "Support for Wacom CTL-672, CTL-472 & XPPen Star G 640" << std::endl << std::endl;

    UniversalTabletDriver driver;

    if (!driver.Initialize()) {
        std::cerr << "Failed to initialize tablet driver" << std::endl;
        std::cout << "Make sure your supported tablet is connected:" << std::endl;
        std::cout << "- Wacom CTL-672" << std::endl;
        std::cout << "- Wacom CTL-472 (One by Wacom Small)" << std::endl;
        std::cout << "- XPPen Star G 640" << std::endl;
        system("pause");
        return 1;
    }

    // Run configuration menu
    driver.RunConfiguration();

    // Wait for driver to exit
    driver.WaitForExit();

    std::cout << "Stopping driver..." << std::endl;
    driver.Stop();

    return 0;
}