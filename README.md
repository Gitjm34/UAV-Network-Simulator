# UAV-Network-Simulator
<div align="center">

# 🚁 **UAV Network Simulator & Adaptive AI-IDS Framework**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![ROS](https://img.shields.io/badge/ROS-Noetic-brightgreen)](https://www.ros.org/)
[![NS-3](https://img.shields.io/badge/NS--3-3.40-orange)](https://www.nsnam.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-black)](https://fastapi.tiangolo.com/)
[![PX4](https://img.shields.io/badge/PX4-v1.14-blueviolet)](https://px4.io/)
[![License](https://img.shields.io/github/license/YOUR_USERNAME/uav-network-simulator-ids)](LICENSE)

**실시간 UAV 네트워크 시뮬레이터 + **Model-Based RL** 기반 **Adaptive AI-IDS** (GIST/KoreaU Collab)**  
**PX4 SITL + Gazebo + NS-3 Dynamic Shaping + ROS MAVROS + FastAPI 1Hz Pipeline ✅ (2025.10.29)**  

**✨ UAV/IoT/Vehicular/Corporate/Home **Multi-Domain** Zero-Label IDS – **AMAGO + GNN + Contrastive Rewards + GenAug** 🚀**

</div>

## 🎯 **왜 이 연구를 시작하게 됐나? (연구 배경 & 동기)**

**현실 세계 네트워크는 동적이다.** 네트워크 토폴로지가 변하고 (노드 추가/제거), 사용자 행동이 바뀌며, **새로운 공격 (Zero-Day)**이 매일 등장합니다. 

### **기존 IDS의 치명적 한계 (Prior Work Limitations)**
- **고정 데이터셋 의존**: CIC-IDS, NSL-KDD 등 **정적 벤치마크**에서 99% ACC → **실전 70%↓** (Unseen 공격 무탐).
- **특정 도메인/공격 특화**: DDoS만, SQL Injection만 → **UAV/IoT/Vehicular/Corporate/Home** 전환 불가.
- **라벨 필수**: **Unlabeled/Adversarial 트래픽** 처리 불가.
- **네트워크 동역학 무시**: 고정 토폴로지 가정 → **실시간 변화 (e.g., 드론 고도 ↑ → Link Degradation)** 대응 X.

**RL의 약속**: 라벨 없이 **Feedback으로 자율 학습** → **Model-Based RL (AMAGO)**로 **Dynamics 예측 + 적응**.

### **이 프로젝트의 미션 (Our Proposal)**
- **현실 Simulator 구축**: **UAV부터 Heterogeneous Networks**까지 **동적 시뮬**.
- **Single Agent**: **하나의 RL 모델**로 **Multi-Domain IDS**.
- **Zero-Label Magic**: **Contrastive Reward Predictor** (Self-Supervised) + **Generative Aug** (GAN-like 신규 공격 생성).
- **관찰**: **GNN Topology Embedding** + **Packet Tokenizer** (NLP-style).

**최종 목표**: **Unseen 환경/공격에서 Robust Detection** – **Paper Target: Jan 2026 Submit** 📜

**저자**: Byeongchang Kim (GIST, kbc202179@gm.gist.ac.kr) + KoreaU Cybersecurity Team

## ✨ **주요 기능**

| 기능 | 설명 | 상태 |
|------|------|------|
| **🔄 Dynamic UAV Link** | 고도(h) → NS-3 Calc (delay=10+h ms, loss=0.3*h%, rate=6000-40*h kbps) → Real-Time Shaping | ✅ 10/09 |
| **🌉 MAVLink Middleware** | `udp_mw_ns3.py`: QGC ↔ PX4 **Transparent Proxy** | ✅ 10/08 |
| **🤖 ROS Telemetry** | `alt2positions.py`: `/mavros/global_position/rel_alt` → `positions.txt` (1Hz) | ✅ 10/09 |
| **📊 FastAPI Pipeline** | **Push**: 1Hz POST `/ingest` (seq/delay/loss/rate/up/down_bytes)<br>**Pull**: `/obs/latest?k=5` `/obs/seq?since=100` | ✅ 10/17 |
| **🧠 RL-IDS Core** | AMAGO + GNN + Token Embed + Contrastive Reward + GenAug | **11/01 Start** |
| **📈 Monitoring** | Live Bytes/Log + Curl API | ✅ |

## 🏗️ **System Architecture**

### **Mermaid Flow**
```mermaid
graph TB
    QGC[QGC<br/>UDP 14640 ↑]
    PX4[PX4 SITL<br/>14540 ↑ / 14550 ↓<br/>50Hz Telemetry]
    Gazebo[Gazebo Physics]
    
    MW[Middleware<br/>udp_mw_ns3.py<br/>**Shaping**]
    NS3[NS-3<br/>mw-link-metrics<br/>positions.txt → Metrics]
    Alt[ROS<br/>alt2positions.py<br/>1Hz Alt → File]
    
    Collector[FastAPI 8080<br/>POST /ingest<br/>GET /obs/*]
    
    RL[🧠 AMAGO Agent<br/>GNN Topology<br/>Packet Tokens]
    Reward[Contrastive<br/>Self-Reward]
    Gen[GenAI Aug]
    
    QGC --> MW
    MW --> PX4
    PX4 --> MW
    MW --> QGC
    
    Alt --> NS3
    NS3 --> MW
    MW --> Collector
    Collector --> RL
    RL --> Reward
    RL --> Gen
    
    PX4 -.-> Alt
    Gazebo <--> PX4
    
    classDef core fill:#e1f5fe
    classDef ai fill:#e8f5e8
    class QGC,PX4,MW,NS3,Alt,Gazebo core
    class RL,Reward,Gen ai
