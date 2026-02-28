# PCS-2 — SHAREKARO: P2P Communication & File Sharing Platform

**Subject:** Programming of Computer Systems 2 (PCS-2, Y2)
**Institution:** IIT Jodhpur, Dept. of CSE & AI
**GitHub:** https://github.com/ADIITJ/SHARE-KARO

## Problem
Peer-to-peer (P2P) platform for direct device-to-device communication and file transfer over a local Wi-Fi network — without relying on a central server. Prioritizes privacy, speed, and simplicity.

## Tech Stack
- Python
- TCP sockets (file transfer with ACK-based integrity)
- psutil (real-time network monitoring)

## Features
- **P2P Messaging:** Direct secure communication between devices on the same network
- **File Transfer:** TCP-based transfer with acknowledgment for data integrity
- **Wi-Fi Network Monitoring:** Real-time stats — bytes sent/received, anomaly detection
- **Cross-Platform:** Runs on any device with Python

## Architecture
```
Device A                    Device B
   │                            │
   │──── TCP Connection ────────│
   │──── File Chunks + ACK ─────│
   │──── Direct Messages ───────│
   │                            │
   └──── psutil network monitor (each side)
```

## How to Run
```bash
git clone https://github.com/ADIITJ/SHARE-KARO
cd SHAREKARO
pip install psutil
python main.py
```

> Full source code and usage: [github.com/ADIITJ/SHARE-KARO](https://github.com/ADIITJ/SHARE-KARO)
