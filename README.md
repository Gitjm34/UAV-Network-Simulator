# UAV-Network-Simulator
<div align="center">

# 🚁 **UAV Network Simulator & Adaptive AI-IDS Framework**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![ROS](https://img.shields.io/badge/ROS-Noetic-brightgreen)](https://www.ros.org/)
[![NS-3](https://img.shields.io/badge/NS--3-3.40-orange)](https://www.nsnam.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-black)](https://fastapi.tiangolo.com/)
[![PX4](https://img.shields.io/badge/PX4-v1.14-blueviolet)](https://px4.io/)
[![License](https://img.shields.io/github/license/YOUR_USERNAME/uav-network-simulator-ids)](LICENSE)
[![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/uav-network-simulator-ids?style=social)](https://github.com/YOUR_USERNAME/uav-network-simulator-ids)

**실시간 UAV(드론) 네트워크 시뮬레이터 + 적응형 강화학습(RL) 기반 AI 침입탐지시스템(IDS)**  
**PX4 SITL + Gazebo + NS-3 네트워크 왜곡 + ROS MAVROS + FastAPI 데이터 파이프라인 완성 (2025.10.29 기준)**  

**✨ 핵심: 드론 고도(h)에 실시간 연동된 무선 링크 품질 시뮬 (delay/loss/rate) + 다중 도메인(드론/IoT/차량/기업/홈) Zero-Label IDS**

</div>

## 📖 **프로젝트 소개: 왜 이걸 만들었나?**

안녕하세요! 이 프로젝트는 **UAV(드론) 보안 연구**를 위해 개발된 **Full-Stack 시뮬레이션 프레임워크**입니다. 실제 드론 비행처럼 **현실적인 네트워크 환경**을 PC 하나로 재현하고, 그 위에서 **AI-IDS**를 훈련/테스트합니다.

### **현실 문제점**
- **드론 통신**: 지상국(QGC) ↔ 드론(PX4) 간 무선 링크(WiFi/LTE)는 **고도(h)에 따라 지연(delay), 패킷 손실(loss), 대역폭(rate)이 동적으로 변함**.
  - 예: h=50m → delay=60ms, loss=15%, rate=2Mbps (현실 공식 적용)
- **기존 IDS 한계**: 정적 데이터셋(CIC-IDS)에 의존 → **새로운 공격/네트워크 변화**에 취약 (99% ACC → 실전 70%↓)
- **연구 목표**: **하나의 RL 에이전트**로 **UAV/IoT/차량/기업/홈 네트워크** 전 도메인에서 **Unseen 공격 자동 적응**.

### **이 프로젝트의 혁신**
1. **동적 네트워크 시뮬**: NS-3로 무선 링크 "가짜" 구현 → 고도 변화 시 **실시간 Traffic Shaping**.
2. **1Hz 데이터 스트림**: Middleware → FastAPI POST → RL 에이전트 Pull (up/down bytes + metrics).
3. **Model-Based RL (AMAGO)**: **GNN(토폴로지) + Tokenized Packets** + **Contrastive Self-Reward** + **Generative Aug**.
   - **라벨 ZERO**: Unlabeled 트래픽으로도 학습!
4. **확장성**: 단일 UAV → Multi-UAV → Hetero Networks.

**🚀 데모 시나리오**: QGC로 드론 이륙 → 고도 100m 상승 → 링크 품질 급락 → IDS가 "공격?" 탐지!

**📊 성과 (10/29)**: 50Hz Telemetry 무중단 + 1Hz Metrics 수집 **100% 안정** (로그 검증).

## ✨ **주요 기능 상세**

| 기능 | 세부 설명 | 상태 |
|------|-----------|------|
| **🔄 동적 링크 시뮬** | `positions.txt` (1Hz) → NS-3 계산 → Middleware 적용<br>**공식**: delay=10+h ms, loss=0.3×h %, rate=6000-40×h kbps | ✅ 완성 |
| **🌉 MAVLink Proxy** | `udp_mw_ns3.py`: QGC(14640) ↔ PX4(14540/50) **투명 중계** + Shaping | ✅ 10/08 |
| **🤖 ROS Alt Monitor** | `alt2positions.py`: MAVROS `/mavros/global_position/rel_alt` → 파일 Write (1Hz) | ✅ 10/09 |
| **📈 FastAPI Collector** | **Push**: `/ingest` (JSON: seq/delay/loss/rate/up/down_bytes)<br>**Pull**: `/obs/latest?k=5`, `/obs/seq?since=100&limit=50` | ✅ 10/17 |
| **🧠 RL-IDS (WIP)** | AMAGO + GNN + Packet Tokenizer + Contrastive Reward + GAN Aug<br>**Multi-Domain**: UAV→Vehicular→... | 11/01~ |
| **📊 Monitoring** | 실시간 up/down bytes 로그 + curl API | ✅ |

## 🏗️ **시스템 아키텍처 (상세 다이어그램)**
<img width="720" height="540" alt="image" src="https://github.com/user-attachments/assets/edf57ee5-8171-4acf-9552-7fe5402ad19d" />
