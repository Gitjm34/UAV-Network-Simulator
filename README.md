# UAV Network Simulator & Adaptive AI-IDS Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![ROS](https://img.shields.io/badge/ROS-Noetic-brightgreen)](https://www.ros.org/)
[![NS-3](https://img.shields.io/badge/NS--3-3.40-orange)](https://www.nsnam.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-black)](https://fastapi.tiangolo.com/)
[![PX4](https://img.shields.io/badge/PX4-v1.14-blueviolet)](https://px4.io/)
[![License](https://img.shields.io/github/license/YOUR_USERNAME/uav-network-simulator-ids)](LICENSE)

**실시간 UAV 네트워크 시뮬레이터 + Model-Based RL 기반 Adaptive AI-IDS**  
**PX4 SITL + Gazebo + NS-3 Dynamic Shaping + ROS MAVROS + FastAPI 1Hz Pipeline**  
UAV/IoT/Vehicular/Corporate/Home **Multi-Domain** Zero-Label IDS – AMAGO + GNN + Contrastive Rewards + GenAug

## 프로젝트 개요
이 프로젝트는 UAV(드론) 네트워크 환경에서 사이버 공격을 시뮬레이션하고, 네트워크 지표와 드론 텔레메트리 데이터를 실시간으로 수집하는 시뮬레이터를 구축한다. 이를 기반으로 강화학습(RL) 기반의 적응형 침입 탐지 시스템(IDS)을 개발하는 것을 목표로 한다. 기존 IDS의 한계를 극복하기 위해, 다양한 네트워크 도메인(UAV/IoT, Vehicle, Corporation, Home)에서 하나의 AI 에이전트가 미확인 공격을 탐지할 수 있도록 설계함

- **주요 기술 스택**: Python 3.10+, ROS Noetic, NS-3 3.40, FastAPI 0.115, PX4 v1.14, Gazebo, MAVROS.
- **협업**: Gwangju Institute of Science and Technology (GIST) AI Graduate School과 Kyung Hee University, Korea University Cybersecurity Department 공동 연구.
- **GitHub 리포지토리**: [YOUR_USERNAME/uav-network-simulator-ids](https://github.com/YOUR_USERNAME/uav-network-simulator-ids) (YOUR_USERNAME을 실제 사용자명으로 교체하세요).
- **데이터셋**: 실험을 통해 생성된 MAVLink 패킷, 네트워크 지표, 드론 텔레메트리 데이터셋 공개 예정 (정상/비정상 레이블 포함)

## 연구 배경 
**현실 세계 네트워크는 동적이다.** 네트워크 토폴로지가 변하고 (노드 추가/제거), 사용자 행동이 바뀌며, 새로운 공격 (Zero-Day)이 매일 등장함 특히 UAV 네트워크는 고도 변화, 이동성, 무선 간섭으로 인해 지연(delay), 손실(loss), 대역폭(rate) 변동이 빈번하다. 이 프로젝트는 이러한 동적 환경에서 기존 IDS의 한계를 극복하기 위해 시작됨

### 기존 IDS의 치명적 한계 (Prior Work Limitations)
- **고정 데이터셋 의존**: CIC-IDS, NSL-KDD 등 **정적 벤치마크**에서 99% ACC → **실전 70%↓** (Unseen 공격 무탐)
- **특정 도메인/공격 특화**: DDoS만, SQL Injection만 → **UAV/IoT/Vehicular/Corporate/Home** 전환 불가
- **라벨 필수**: **Unlabeled/Adversarial 트래픽** 처리 불가
- **네트워크 동역학 무시**: 고정 토폴로지 가정 → **실시간 변화 (e.g., 드론 고도 ↑ → Link Degradation)** 대응 X
- **RL 기반 IDS 한계**: 기존 RL-IDS는 시뮬레이션 환경이 제한적이며, 수동 보상 설계에 의존. AMAGO 같은 모델 기반 RL로 동적 예측을 강화해야 함
**Reinforcement learning**: 라벨 없이 **Feedback으로 자율 학습** → Model-Based RL (AMAGO)로 **Dynamics 예측 + 적응**

### 프로젝트의 미션 (Our Proposal)
- **현실 반영 Simulator 구축**: **UAV부터 Heterogeneous Networks**까지 **동적 시뮬**. PX4 SITL + NS-3로 고도 기반 네트워크 품질 동적 적용 (delay=10+h ms, loss=0.3*h %, rate=6000-40*h kbps)
- **Single Agent**: **하나의 RL 모델**로 **Multi-Domain IDS**. GNN으로 네트워크 토폴로지 임베딩 + 패킷 토큰화(NLP 스타일)로 트래픽 분석
- **Zero-Label Magic**: **Contrastive Reward Predictor** (Self-Supervised) + **Generative Aug** (GAN-like 신규 공격 생성). 라벨 없이 적응 학습
- **관찰**: **GNN Topology Embedding** + **Packet Tokenizer** (NLP-style)
- **시뮬레이션 공격 지원**: DoS (과도 트래픽으로 지연/손실 증가), Heartbeat Drop (연결 끊김 유발). attackctl.py로 제어 (e.g., dos 20 800 30 3 또는 hb 15 0.6)
**최종 목표**: **Unseen 환경/공격에서 Robust Detection** – **Paper Target: Jan 2026 Submit** 📜 (IEEE Transactions on Information Forensics and Security 또는 유사 저널)
**데이타셋 생성**: 실험을 통한 MAVLink, Sensor Dataset 생성 및 배포 (PCAPNG 형식 RAW 패킷 + JSON 요약, CSV 변환 지원)

## 주요 기능
| 기능 | 설명 | 상태 | 날짜 |
|------|------|------|------|
| **Dynamic UAV Link** | 고도(h) → NS-3 Calc (delay=10+h ms, loss=0.3*h%, rate=6000-40*h kbps) → Real-Time Shaping | ✅ | 10/09 |
| **MAVLink Middleware** | `udp_mw_ns3.py`: QGC ↔ PX4 **Transparent Proxy** + 네트워크 지표 집계 (up/down_bytes, seq) | ✅ | 10/08 |
| **ROS Telemetry** | `alt2positions.py`: `/mavros/global_position/rel_alt` → `positions.txt` (1Hz) + `ros_extra_pusher.py`로 추가 텔레메트리 (고도, 속도, GNSS 상태, heartbeat gap, MAVLink Hz) 전송 | ✅ | 10/09 |
| **FastAPI Pipeline** | **Push**: 1Hz POST `/ingest` (seq/delay/loss/rate/up/down_bytes) + `/ingest_extra` (텔레메트리)<br>**Pull**: `/obs/latest?k=5` `/obs/seq?since=100` (합본 조회) | ✅ | 10/17 |
| **RAW Packet Capture** | `tcpdump` + `tshark`로 MAVLink/UDP 트래픽 PCAPNG 저장 + CSV 변환 (시간, 출발지, 목적지, 포트, 길이, hex 바디) | ✅ | 10/22 |
| **Attack Simulation** | DoS (대역폭 점유) + Heartbeat Drop (메시지 드롭) 시뮬레이션, `attackctl.py`로 제어 | ✅ | 11/02 |
| **RL-IDS Core** | AMAGO + GNN + Token Embed + Contrastive Reward + GenAug | 🔄 진행 중 | 11/01 Start |
| **Monitoring** | Live Bytes/Log + Curl API + JSON 로그 (~/.uav_ids/flow.jsonl) | ✅ | 10/17 |
| **Dataset Generation** | 정상/비정상 데이터셋 자동 생성 (PCAP + JSON + CSV) | ✅ | 10/22 |

## 🏗️ System Architecture
![System Architecture](https://github.com/user-attachments/assets/423a1bef-9a82-408b-bc0d-d2bea4e28ab5)

- **PX4 SITL (Gazebo)**: 가상 드론 autopilot. MAVLink #0: UDP 14540 (server), MAVLink #1: UDP Client → 127.0.0.1:14550 (to MW).
- **Middleware (udp_mw_ns3.py)**: Receives: 14640 (from QGC) 14550 (from PX4). Forwards to PX4 14540 (FCU) to QGC inbound (e.g.,1550). Apply ns-3 delay/loss/rate. Logs: up_bytes, down_bytes, seq. POST /ingest -> Collector.
- **QGroundControl**: Connect to host: 127.0.0.1:14640. Listen: OFF. Inbound from MW: dynamic (e.g., ~1550).
- **MAVROS**: Bind: (e.g.)14558. Send -> 127.0.0.1:14556. alti2positions.py Writes Position.txt (1Hz). ros_extra_pusher.py POST /ingest_extra -> Collector.
- **ns-3 (mw-link-metrics)**: Reads Positions.txt. Calculates delay/loss/rate (for shaping).
- **Collector (Fast API)**: POST /ingest (Network). POST /ingest_extra (Drone Telemetry). GET /obs/latest, GET /obs/seq. Port 8080.
- **데이터 흐름**: 드론 고도 변화 → positions.txt 업데이트 → ns-3 계산 → 미들웨어 적용 → Collector 수집. RAW 패킷 캡처(tcpdump) + 변환(tshark) 지원.

## 🛠️ 설치 가이드 (Dependencies & Setup)
Ubuntu 20.04 (ARM64) 기반으로 테스트됨. ROS Noetic, NS-3, PX4 등 설치 필요.

### 1. 기본 패키지 설치
```bash
sudo apt-get update
sudo apt-get install -y python3-pip lsof netcat-openbsd tcpdump tshark git cmake libxml2-utils
python3 -m pip install --user tqdm ecdsa numpy scipy pandas matplotlib sympy requests fastapi uvicorn pydantic pymavlink
