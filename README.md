# UltraTabletDriver
This is really good tablet driver mostly for playing Osu!

It is in C++... That means it is extremly fast and low latency - Faster than OpenTabletDriver

## Support:
- A huge thank you to an Osu! streamer Rumpowl who is helping me with testing and ideas: twitch.tv/rumpowl

## Features:
- Prediction: Predicts where you most likely want to move your cursor (might add snaps, configruable)
- Smoothing: Makes your cursor move smoother (It increases latency when higher smoothing, configurable)
- Jitter reducer (If the movement is below a given threshold it will prevent the cursor from moving. configurable)
- Custom tablet area (size, location, rotation)
- Monitor switching
- Simple GUI


## Current known issues:
- The Height and Width are not swapping when rotating the area

## Supported tablets:
- Wacom CTL-672 (One By Wacom Medium) - the only tested and 100% working tablet
- Wacom CTL-472 (One By Wacom Small)
- XP Pen Star G640

## Minimum requirements:
- CPU with at least 4 threads recommended - The driver is CPU based + uses multithreadning
- Windows 10/11 - tested on 11 only
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
