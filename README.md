# UltraTabletDriver
This is really good tablet driver mostly for playing Osu!

It is in C++... That means it is extremly fast and low latency.

## Features:
- Prediction: Predicts where you most likely want to move your cursor (configruable)
- Interpolation: Generates cursor steps to fill gaps between every tablet update (might increase input latency, configurable)
- Custom tablet area (size, location, rotation)
- Monitor switching


## Current issues:
- Interpolation is slow, unstable - WIP

## Supported tablets:
- Wacom CTL-672 (One By Wacom Medium) - the only tested and 100% working tablet
- Wacom CTL-472 (One By Wacom Small)
- XP Pen Star G640

## Minimum requirements:
- CPU with at least 4 threads
- One of the supported tablets

Note: It might not run well on bad hardware.

## How to run:
Download the latest UltraTabletDriver.exe file from Releases.

Download OpenTabletDriver and run it as administrator.

Close OpenTabletDriver (make sure to close it in system tray too)

Connect your tablet to PC and run UltraTabletDriver.exe as Administrator

Every time you connect/reconnect your tablet to PC, you have to repeat these steps else the driver won't work

## Common issues:
- The tablet is not detected by the program - Try closing/terminating every other driver... If it still doesn't work, try running OpenTabletDriver as Administrator and than close it and try again (UltraTabletDriver in current state can't bypass default Windows tablet driver)
- I can't move the cursor using the pen when I am playing Osu! - You have to disable Raw Input in Osu! settings (Raw input support - work in progress)
