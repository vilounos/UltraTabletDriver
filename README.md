# UltraTabletDriver
This is really good tablet driver mostly for playing Osu!

It is in C++... That means it is extremly fast and low latency - Faster than OpenTabletDriver

If you have any issues, questions or want to add support for your tablet or anything else, you can add me and contact me on Discord: vilounos

## Support:
- A huge thank you to an Osu! streamer Rumpowl who is helping me with testing and ideas: twitch.tv/rumpowl

## Features:
- Prediction: Predicts where you most likely want to move your cursor (might add snaps, configruable)
- Smoothing: Makes your cursor move smoother (It increases latency when higher smoothing, configurable)
- Jitter reducer (If the movement is below a given threshold it will prevent the cursor from moving. configurable)
- Advanced logging/monitoring system
- Custom tablet area (size, location, rotation)
- Monitor switching
- Simple GUI (still some visual bugs)

## Current known issues:
- The Height and Width are not swapping when rotating the area
- GUI blinking

## Supported tablets:
- Wacom CTL-672 (One By Wacom Medium) - tested
- Wacom CTL-472 (One By Wacom Small) - tested
- XP Pen Star G640 - not tested but should work

-  If you want to add support for your tablet, follow instruction in this repository: [GetRawData](https://github.com/vilounos/getrawdata-tablet)

## Minimum requirements:
- CPU with at least 4 threads recommended - The driver is CPU based + uses multithreadning
- Windows 10/11 - tested on 11 only
- One of the supported tablets

Note: It might not run well on bad hardware.

## How to run:
Download the latest [UltraTabletDriver.exe](https://github.com/vilounos/UltraTabletDriver/releases) file from Releases.

Download [OpenTabletDriver](https://github.com/OpenTabletDriver/OpenTabletDriver) and run it as administrator.

Close [OpenTabletDriver](https://github.com/OpenTabletDriver/OpenTabletDriver) (make sure to close it in system tray too)

Connect your tablet to PC and run [UltraTabletDriver.exe](https://github.com/vilounos/UltraTabletDriver/releases) as Administrator

Every time you connect/reconnect your tablet to PC, you have to repeat these steps else the driver won't work

## Common issues:
- The tablet is not detected by the program - Try closing/terminating every other driver... If it still doesn't work, try running [OpenTabletDriver](https://github.com/OpenTabletDriver/OpenTabletDriver) as Administrator and than close it and try again ([UltraTabletDriver](https://github.com/vilounos/UltraTabletDriver/releases) in current state can't bypass default Windows tablet driver)
