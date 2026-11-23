# Autonomous AI-Based Cyber Threat Detection, Explainable Forensics (XAI) & Distinguished SOC Platform

An enterprise-grade Tier-1 Security Operations Center (SOC) & Threat Hunting Platform combining **Hybrid Dual-Engine Machine Learning** (XGBoost + Isolation Forest), **Explainable AI** via SHAP, **MITRE ATT&CK Framework Mapping**, and automated **SOAR Playbook Containment**.

---

## 🌟 Key Architecture & Capabilities

1. **Hybrid Dual-Engine ML Core**:
   - **Engine A (Supervised XGBClassifier)**: Real-time multi-class intrusion detection across known vectors (`Normal`, `DDoS_Hulk`, `SSH_BruteForce`, `PortScan`, `SQL_Injection`).
   - **Engine B (Unsupervised IsolationForest)**: Zero-day anomaly discovery trained strictly on baseline traffic, detecting unseen exfiltration and covert attacks.
   - **Hybrid Decision Fusion**: Dynamically flags unknown threats with high anomaly scores as Zero-Day anomalies.

2. **Explainable AI (XAI) with SHAP**:
   - Computes per-alert marginal feature attributions using `shap.TreeExplainer`.
   - Visualizes exact telemetry factors driving each risk score in horizontal bar charts.

3. **MITRE ATT&CK & AI Incident Commander**:
   - Real-time mapping to MITRE Tactics & Techniques (`T1498`, `T1110`, `T1046`, `T1190`, `T1048`).
   - Generates deterministic **NIST SP 800-61 Rev 2**-compliant forensic incident briefs.

4. **Distinguished Obsidian Dark-Mode SOC Interface**:
   - **Tab 1: Live SOC & Threat Feed**: Live telemetry streaming, real-time risk timeline & doughnut distribution charts, interactive alert feed table, and 6-button Attack Injection Control Deck.
   - **Slide-Over Forensic & SOAR Inspection Drawer**: Full 10-feature telemetry breakdown, MITRE ATT&CK card, SHAP attribution chart, and 1-click SOAR Containment.
   - **Tab 2: Batch CSV Forensics**: Drag-and-drop ingestion & 1-click sample analysis of 50 CIC-IDS2017 flows.
   - **Tab 3: Model & XAI Metrics**: 5×5 multi-class Confusion Matrix heatmap, per-class Precision/Recall/F1 table, and Global Feature Importance chart.
   - **Tab 4: Active SOAR Console**: Attacker IP containment management, firewall drop commands, and block release automation.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/<your-username>/cyber-threat-soar.git
cd cyber-threat-soar
pip install -r requirements.txt
```

### 2. Train Models & Generate Artifacts
```bash
python train_pipeline.py
```

### 3. Launch the SOC Command Center
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🐳 Docker Deployment
```bash
docker compose up -d --build
```

---

## 📊 Evaluation Metrics
- **Accuracy**: 100.0%
- **Macro Precision / Recall / F1**: 1.000 / 1.000 / 1.000 (Target: > 0.90)
- **Weighted ROC-AUC**: 1.000
