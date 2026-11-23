"""
Module 3: XAI Forensics, MITRE ATT&CK & SOAR Engine (xai_soar_engine.py)
Implements Hybrid Decision Fusion, Real-time SHAP Attribution, MITRE ATT&CK Mapping,
SOAR Playbook Rule Generation, and AI Incident Commander Summaries.
"""

import os
import joblib
import numpy as np
import shap
import httpx
from data_generator import FEATURE_NAMES

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

MITRE_PLAYBOOKS = {
    "Normal": {
        "tactic": "Benign Operational",
        "technique_id": "None",
        "technique_name": "Standard Network Telemetry",
        "severity": "LOW",
        "base_risk": 5.0,
        "mitigation": "No active mitigation required. Flow matches benign operational baseline.",
    },
    "DDoS_Hulk": {
        "tactic": "Impact",
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "severity": "CRITICAL",
        "base_risk": 88.0,
        "mitigation": "iptables -I INPUT -s {src_ip} -j DROP && rate-limit --limit 50/s --burst 100",
    },
    "SSH_BruteForce": {
        "tactic": "Credential Access",
        "technique_id": "T1110",
        "technique_name": "Brute Force Authentication",
        "severity": "HIGH",
        "base_risk": 78.0,
        "mitigation": "iptables -I INPUT -p tcp --dport 22 -s {src_ip} -j DROP && fail2ban-client set sshd banip {src_ip}",
    },
    "PortScan": {
        "tactic": "Discovery",
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "severity": "MEDIUM",
        "base_risk": 55.0,
        "mitigation": "iptables -I INPUT -s {src_ip} -j DROP && ipset add scanner_blacklist {src_ip}",
    },
    "SQL_Injection": {
        "tactic": "Initial Access",
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "severity": "CRITICAL",
        "base_risk": 94.0,
        "mitigation": "iptables -I INPUT -s {src_ip} -j DROP && modsecurity-waf-rule --block-ip {src_ip}",
    },
    "ZeroDay_Anomaly": {
        "tactic": "Exfiltration",
        "technique_id": "T1048",
        "technique_name": "Exfiltration Over Alternative Protocol (Zero-Day)",
        "severity": "CRITICAL",
        "base_risk": 95.0,
        "mitigation": "iptables -I FORWARD -s {src_ip} -j DROP && isolate-host --quarantine {src_ip} --snort-pcap-capture",
    },
}


class XAISOAREngine:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.load_models()

    def load_models(self):
        xgb_path = os.path.join(self.artifacts_dir, "xgb_model.joblib")
        iso_path = os.path.join(self.artifacts_dir, "iso_forest.joblib")
        scaler_path = os.path.join(self.artifacts_dir, "scaler.joblib")
        le_path = os.path.join(self.artifacts_dir, "label_encoder.joblib")

        if not all(os.path.exists(p) for p in [xgb_path, iso_path, scaler_path, le_path]):
            raise FileNotFoundError("Trained models not found. Run train_pipeline.py first.")

        self.xgb_model = joblib.load(xgb_path)
        self.iso_forest = joblib.load(iso_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(le_path)

        # Initialize shap TreeExplainer once at startup
        print("[*] Initializing SHAP TreeExplainer on XGBoost model...")
        self.explainer = shap.TreeExplainer(self.xgb_model)
        print("[+] SHAP TreeExplainer initialized successfully.")

    def analyze_flow(self, flow: dict) -> dict:
        """
        Executes Hybrid Decision Fusion, SHAP attribution, MITRE mapping,
        and generates an enriched alert object.
        """
        raw_features = [float(flow[k]) for k in FEATURE_NAMES]
        vec = np.array([raw_features])
        vec_scaled = self.scaler.transform(vec)

        # 1. Engine A: Supervised XGBoost
        proba_a = self.xgb_model.predict_proba(vec_scaled)[0]
        pred_idx = int(np.argmax(proba_a))
        conf_a = float(proba_a[pred_idx])
        pred_label_a = str(self.label_encoder.inverse_transform([pred_idx])[0])

        # 2. Engine B: Unsupervised Isolation Forest Anomaly Detection
        dec_func = float(self.iso_forest.decision_function(vec_scaled)[0])
        # Normalized anomaly score 0.0 - 1.0 (higher = more anomalous)
        anomaly_score = float(np.clip(0.50 - dec_func * 2.2, 0.0, 1.0))

        # 3. Hybrid Decision Fusion
        # If Engine A predicts Normal (or confidence < 0.65) while Engine B anomaly score > 0.68,
        # classify as ZeroDay_Anomaly flagged by Engine B (Isolation Forest).
        if (pred_label_a == "Normal" or conf_a < 0.65) and anomaly_score > 0.68:
            threat_label = "ZeroDay_Anomaly"
            engine_source = "Isolation Forest"
            decision_confidence = anomaly_score
            is_zero_day = True
        else:
            threat_label = pred_label_a
            engine_source = "XGBoost"
            decision_confidence = conf_a
            is_zero_day = False

        # 4. SHAP Attribution (Top 4 contributing features)
        shap_vals = self.explainer(vec_scaled)
        # For Zero-day or Normal, inspect deviation from Normal baseline
        if threat_label == "ZeroDay_Anomaly":
            normal_idx = int(self.label_encoder.transform(["Normal"])[0])
            impacts = -shap_vals.values[0, :, normal_idx]
        else:
            class_idx = int(self.label_encoder.transform([pred_label_a])[0])
            impacts = shap_vals.values[0, :, class_idx]

        feature_attributions = []
        for i, feat in enumerate(FEATURE_NAMES):
            feature_attributions.append({
                "feature": feat,
                "value": flow[feat],
                "shap_impact": round(float(impacts[i]), 4),
            })

        # Sort by absolute SHAP impact descending and pick top 4
        top_shap_features = sorted(
            feature_attributions, key=lambda x: abs(x["shap_impact"]), reverse=True
        )[:4]

        # 5. MITRE ATT&CK & SOAR Playbook Mapping
        playbook = MITRE_PLAYBOOKS.get(threat_label, MITRE_PLAYBOOKS["Normal"])
        src_ip = flow.get("src_ip", "0.0.0.0")
        dst_port = flow.get("dst_port", 80)
        mitigation_cmd = playbook["mitigation"].format(src_ip=src_ip)

        # 6. Dynamic Risk Score (0 - 100)
        if threat_label == "Normal":
            risk_score = round(anomaly_score * 18.0, 1)
        elif threat_label == "ZeroDay_Anomaly":
            risk_score = round(92.0 + (anomaly_score - 0.68) * 25.0, 1)
            risk_score = min(100.0, max(92.0, risk_score))
        elif threat_label == "DDoS_Hulk":
            risk_score = round(85.0 + 13.0 * min(1.0, conf_a), 1)
        elif threat_label == "SSH_BruteForce":
            risk_score = round(75.0 + 15.0 * min(1.0, conf_a), 1)
        elif threat_label == "PortScan":
            risk_score = round(50.0 + 20.0 * min(1.0, conf_a), 1)
        elif threat_label == "SQL_Injection":
            risk_score = round(90.0 + 9.0 * min(1.0, conf_a), 1)
        else:
            risk_score = 50.0

        risk_score = min(100.0, max(0.0, risk_score))

        # Status: ACTIVE / CONTAINED / BENIGN
        initial_status = "BENIGN" if threat_label == "Normal" else "ACTIVE"

        # 7. AI Incident Commander Summary (Deterministic fallback + LLM integration)
        incident_summary = self._generate_incident_summary(
            src_ip=src_ip,
            dst_port=dst_port,
            threat_label=threat_label,
            engine_source=engine_source,
            risk_score=risk_score,
            confidence=decision_confidence,
            top_shap=top_shap_features,
            playbook=playbook,
            mitigation_cmd=mitigation_cmd,
        )

        return {
            "event_id": flow.get("event_id"),
            "timestamp": flow.get("timestamp"),
            "src_ip": src_ip,
            "dst_ip": flow.get("dst_ip"),
            "protocol": flow.get("protocol", "TCP"),
            "dst_port": dst_port,
            "threat_label": threat_label,
            "engine_source": engine_source,
            "confidence": round(decision_confidence, 4),
            "anomaly_score": round(anomaly_score, 4),
            "is_zero_day": is_zero_day,
            "risk_score": risk_score,
            "status": initial_status,
            "mitre_tactic": playbook["tactic"],
            "mitre_technique_id": playbook["technique_id"],
            "mitre_technique_name": playbook["technique_name"],
            "severity": playbook["severity"],
            "mitigation_cmd": mitigation_cmd,
            "top_shap_features": top_shap_features,
            "all_features": {k: flow[k] for k in FEATURE_NAMES},
            "ai_incident_summary": incident_summary,
        }

    def _generate_incident_summary(
        self,
        src_ip: str,
        dst_port: int,
        threat_label: str,
        engine_source: str,
        risk_score: float,
        confidence: float,
        top_shap: list,
        playbook: dict,
        mitigation_cmd: str,
    ) -> str:
        """
        NIST SP 800-61 Rev 2-aligned forensic summary.
        Uses deterministic generation with dynamic flow metrics.
        """
        if threat_label == "Normal":
            return (
                f"NIST SP 800-61 Benign Verification: Flow from {src_ip} targeting port {dst_port} "
                f"matches verified benign telemetry profiles. Engine: {engine_source} (Risk Score: {risk_score}/100). "
                f"Traffic parameters are within operational tolerance baselines. Status: BENIGN."
            )

        shap_details = ", ".join([
            f"{s['feature']}={s['value']} (impact: {s['shap_impact']:+0.2f})"
            for s in top_shap
        ])

        summary = (
            f"NIST SP 800-61 Rev 2 Incident Brief: Threat vector detected from host {src_ip} "
            f"targeting service port {dst_port}. Detection Engine: {engine_source} classified this flow as "
            f"'{threat_label}' with dynamic Risk Score {risk_score}/100 (Confidence: {confidence * 100:.1f}%). "
            f"MITRE ATT&CK Mapping: {playbook['technique_id']} - {playbook['technique_name']} [{playbook['tactic']}]. "
            f"Key forensic feature deviations: {shap_details}. "
            f"SOAR Playbook Containment: Execute '{mitigation_cmd}' to neutralize egress/ingress vectors."
        )
        return summary


if __name__ == "__main__":
    from data_generator import generate_single_flow
    engine = XAISOAREngine()
    print("\n--- Testing Normal Flow ---")
    res_normal = engine.analyze_flow(generate_single_flow("Normal"))
    print(res_normal["threat_label"], res_normal["engine_source"], "Risk:", res_normal["risk_score"])

    print("\n--- Testing DDoS Flow ---")
    res_ddos = engine.analyze_flow(generate_single_flow("DDoS_Hulk"))
    print(res_ddos["threat_label"], res_ddos["engine_source"], "Risk:", res_ddos["risk_score"])
    print("Top SHAP:", res_ddos["top_shap_features"])

    print("\n--- Testing ZeroDay Flow ---")
    res_zd = engine.analyze_flow(generate_single_flow("ZeroDay_Exfiltration"))
    print(res_zd["threat_label"], res_zd["engine_source"], "Risk:", res_zd["risk_score"])
    print("Top SHAP:", res_zd["top_shap_features"])
    print("AI Summary:", res_zd["ai_incident_summary"])
