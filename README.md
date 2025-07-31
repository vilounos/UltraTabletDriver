# UltraTabletDriver
This is really good tablet driver mostly for playing Osu!
It is made in C++... That means it is extremly fast and low latency.
It also has a configurable "Prediction" function - It "removes" the hardware latency.

## Supported tablets:
- Wacom CTL-672 (One By Wacom Medium) - the only tested and 100% working tablet
- Wacom CTL-472 (One By Wacom Small)
- XP Pen Star G640

## How to run:
Download the latest UltraTabletDriver.exe file from Releases.
Connect your tablet to PC and run UltraTabletDriver.exe as Administrator

## Common issues:
The tablet is not detected by the program - Try closing/terminating every other driver... If it still doesn't work, try running OpenTabletDriver as Administrator and than close it and try again (UltraTabletDriver in current state can't bypass default Windows tablet driver)
I can't move the cursor using the pen when I am playing Osu! - You have to disable Raw Input in Osu! settings
