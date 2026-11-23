"""
Module 1: CIC-IDS2017 Telemetry Generator (data_generator.py)
Generates realistic network flow telemetry and training datasets following the CIC-IDS2017 benchmark profile.
"""

import os
import uuid
import random
from datetime import datetime
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "dst_port",
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "flow_iat_mean",
    "syn_flag_count",
    "ack_flag_count",
    "pkt_len_variance",
]

KNOWN_CLASSES = [
    "Normal",
    "DDoS_Hulk",
    "SSH_BruteForce",
    "PortScan",
    "SQL_Injection",
]

ALL_CLASSES = KNOWN_CLASSES + ["ZeroDay_Exfiltration"]

COMMON_SRC_IPS = {
    "Normal": ["192.168.1.105", "192.168.1.112", "10.0.0.45", "172.16.0.22"],
    "DDoS_Hulk": ["185.220.101.5", "185.220.101.9", "45.142.122.18", "194.26.29.112"],
    "SSH_BruteForce": ["203.0.113.88", "198.51.100.19", "218.92.0.210", "112.85.42.91"],
    "PortScan": ["45.33.32.156", "185.191.171.12", "79.124.62.86", "162.243.128.51"],
    "SQL_Injection": ["198.51.100.42", "194.135.25.33", "103.203.57.18", "91.240.118.172"],
    "ZeroDay_Exfiltration": ["193.161.193.99", "185.156.73.44", "104.244.76.13", "194.67.210.5"],
}

COMMON_DST_IPS = ["192.168.1.10", "192.168.1.20", "10.0.0.1", "172.16.0.100"]


def _random_ip(subnet=""):
    if subnet:
        return f"{subnet}.{random.randint(2, 254)}"
    return f"{random.randint(11, 220)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(2, 254)}"


def _sample_features_for_class(attack_type: str) -> dict:
    """Generate 10 numerical ML features based on cybersecurity telemetry semantics."""
    if attack_type == "Normal":
        dst_port = int(random.choices([80, 443, 8080, 53, 21, 8443, 3306], weights=[0.4, 0.4, 0.08, 0.05, 0.03, 0.02, 0.02])[0])
        flow_duration = float(np.clip(np.random.normal(35000, 9000), 4000, 120000))
        total_fwd_packets = int(np.clip(np.random.normal(12, 4), 3, 45))
        total_bwd_packets = int(np.clip(np.random.normal(14, 5), 3, 50))
        flow_bytes_per_sec = float(np.clip(np.random.normal(8500, 2500), 500, 30000))
        flow_packets_per_sec = float(np.clip(np.random.normal(80, 22), 10, 280))
        flow_iat_mean = float(np.clip(np.random.normal(1800, 450), 250, 5500))
        syn_flag_count = int(random.choices([0, 1, 2], weights=[0.05, 0.92, 0.03])[0])
        ack_flag_count = int(np.clip(np.random.normal(15, 4), 2, 45))
        pkt_len_variance = float(np.clip(np.random.normal(180, 50), 20, 500))

    elif attack_type == "DDoS_Hulk":
        dst_port = int(random.choices([80, 443, 8080], weights=[0.6, 0.35, 0.05])[0])
        flow_duration = float(np.clip(np.random.normal(8000, 2200), 1200, 22000))
        total_fwd_packets = int(np.clip(np.random.normal(130, 25), 50, 260))
        total_bwd_packets = int(np.clip(np.random.normal(1, 0.8), 0, 4))
        flow_bytes_per_sec = float(np.clip(np.random.normal(700000, 140000), 280000, 1800000))
        flow_packets_per_sec = float(np.clip(np.random.normal(10500, 2200), 3500, 32000))
        flow_iat_mean = float(np.clip(np.random.normal(45, 12), 6, 110))
        syn_flag_count = int(np.clip(np.random.normal(20, 5), 6, 45))
        ack_flag_count = int(np.clip(np.random.normal(1, 0.8), 0, 4))
        pkt_len_variance = float(np.clip(np.random.normal(38, 12), 4, 110))

    elif attack_type == "SSH_BruteForce":
        dst_port = 22 if random.random() < 0.98 else 2222
        flow_duration = float(np.clip(np.random.normal(26000, 6000), 9000, 55000))
        total_fwd_packets = int(np.clip(np.random.normal(25, 5), 14, 48))
        total_bwd_packets = int(np.clip(np.random.normal(22, 5), 12, 42))
        flow_bytes_per_sec = float(np.clip(np.random.normal(7200, 1800), 2200, 16000))
        flow_packets_per_sec = float(np.clip(np.random.normal(340, 70), 140, 650))
        flow_iat_mean = float(np.clip(np.random.normal(420, 100), 120, 850))
        syn_flag_count = int(np.clip(np.random.normal(2, 0.7), 1, 4))
        ack_flag_count = int(np.clip(np.random.normal(24, 4), 12, 45))
        pkt_len_variance = float(np.clip(np.random.normal(320, 70), 130, 600))

    elif attack_type == "PortScan":
        dst_port = int(random.choice([21, 22, 23, 25, 53, 80, 110, 135, 139, 443, 445, 1433, 3306, 3389, 5432, 8080]))
        flow_duration = float(np.clip(np.random.normal(550, 160), 60, 2200))
        total_fwd_packets = int(np.clip(np.random.normal(2, 0.7), 1, 4))
        total_bwd_packets = int(np.clip(np.random.normal(0.2, 0.4), 0, 2))
        flow_bytes_per_sec = float(np.clip(np.random.normal(750, 240), 120, 2200))
        flow_packets_per_sec = float(np.clip(np.random.normal(3600, 850), 1300, 11000))
        flow_iat_mean = float(np.clip(np.random.normal(68, 20), 12, 250))
        syn_flag_count = int(np.clip(np.random.normal(2, 0.5), 1, 3))
        ack_flag_count = int(np.clip(np.random.normal(0.2, 0.4), 0, 1))
        pkt_len_variance = float(np.clip(np.random.normal(6, 3), 0, 28))

    elif attack_type == "SQL_Injection":
        dst_port = int(random.choices([80, 443, 3306, 8080], weights=[0.45, 0.35, 0.15, 0.05])[0])
        flow_duration = float(np.clip(np.random.normal(44000, 10000), 14000, 85000))
        total_fwd_packets = int(np.clip(np.random.normal(20, 5), 9, 42))
        total_bwd_packets = int(np.clip(np.random.normal(24, 6), 9, 50))
        flow_bytes_per_sec = float(np.clip(np.random.normal(42000, 9500), 18000, 90000))
        flow_packets_per_sec = float(np.clip(np.random.normal(240, 55), 90, 480))
        flow_iat_mean = float(np.clip(np.random.normal(1150, 280), 350, 2800))
        syn_flag_count = 1
        ack_flag_count = int(np.clip(np.random.normal(22, 5), 8, 48))
        pkt_len_variance = float(np.clip(np.random.normal(2900, 650), 1350, 6800))

    elif attack_type == "ZeroDay_Exfiltration":
        dst_port = int(random.choice([8443, 9001, 31337, 4444, 49152, 55555]))
        flow_duration = float(np.clip(np.random.normal(155000, 30000), 75000, 380000))
        total_fwd_packets = int(np.clip(np.random.normal(195, 35), 90, 420))
        total_bwd_packets = int(np.clip(np.random.normal(2.5, 1.0), 1, 6))
        flow_bytes_per_sec = float(np.clip(np.random.normal(420000, 80000), 180000, 850000))
        flow_packets_per_sec = float(np.clip(np.random.normal(980, 200), 380, 2100))
        flow_iat_mean = float(np.clip(np.random.normal(680, 150), 200, 1500))
        syn_flag_count = 1
        ack_flag_count = int(np.clip(np.random.normal(9, 2.5), 3, 20))
        pkt_len_variance = float(np.clip(np.random.normal(3400, 750), 1500, 6200))

    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    return {
        "dst_port": dst_port,
        "flow_duration": round(flow_duration, 2),
        "total_fwd_packets": total_fwd_packets,
        "total_bwd_packets": total_bwd_packets,
        "flow_bytes_per_sec": round(flow_bytes_per_sec, 2),
        "flow_packets_per_sec": round(flow_packets_per_sec, 2),
        "flow_iat_mean": round(flow_iat_mean, 2),
        "syn_flag_count": syn_flag_count,
        "ack_flag_count": ack_flag_count,
        "pkt_len_variance": round(pkt_len_variance, 2),
    }


def generate_single_flow(attack_type: str = None) -> dict:
    """
    Generates a realistic flow for any of the 5 known classes
    OR the 6th unseen vector ZeroDay_Exfiltration.
    """
    if not attack_type:
        attack_type = random.choice(ALL_CLASSES)

    features = _sample_features_for_class(attack_type)

    src_pool = COMMON_SRC_IPS.get(attack_type, ["10.10.10.10"])
    src_ip = random.choice(src_pool) if random.random() < 0.75 else _random_ip()
    dst_ip = random.choice(COMMON_DST_IPS)
    protocol = "TCP"

    event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    flow = {
        "event_id": event_id,
        "timestamp": timestamp,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "protocol": protocol,
        **features,
        "actual_label": attack_type,
    }
    return flow


def generate_training_data(n_samples: int = 6000) -> pd.DataFrame:
    """
    Produces balanced, realistic traffic across 5 known classes
    (Normal, DDoS_Hulk, SSH_BruteForce, PortScan, SQL_Injection)
    with realistic Gaussian overlap so validation accuracy sits around 94%-98%.
    """
    samples_per_class = n_samples // len(KNOWN_CLASSES)
    rows = []

    for label in KNOWN_CLASSES:
        for _ in range(samples_per_class):
            features = _sample_features_for_class(label)
            src_pool = COMMON_SRC_IPS.get(label, ["192.168.1.100"])
            src_ip = random.choice(src_pool) if random.random() < 0.7 else _random_ip()
            dst_ip = random.choice(COMMON_DST_IPS)
            row = {
                "event_id": f"EVT-{uuid.uuid4().hex[:8].upper()}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "protocol": "TCP",
                **features,
                "label": label,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def export_sample_test_flows(filepath: str = "artifacts/sample_test_flows.csv", n_flows: int = 50) -> str:
    """
    Automatically saves sample test flows (50 mixed flows including 4 zero-day anomalies)
    for immediate testing in the Batch Forensics tab.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # 4 Zero-day, 20 Normal, 8 DDoS, 6 SSH, 6 PortScan, 6 SQLi
    distribution = (
        ["ZeroDay_Exfiltration"] * 4
        + ["Normal"] * 20
        + ["DDoS_Hulk"] * 8
        + ["SSH_BruteForce"] * 6
        + ["PortScan"] * 6
        + ["SQL_Injection"] * 6
    )
    random.shuffle(distribution)

    flows = [generate_single_flow(attack) for attack in distribution[:n_flows]]
    df = pd.DataFrame(flows)
    df.to_csv(filepath, index=False)
    print(f"[+] Sample test flows exported: {filepath} ({len(df)} rows)")
    return filepath


if __name__ == "__main__":
    df_train = generate_training_data(6000)
    print(f"Training data generated: {df_train.shape}")
    print(df_train["label"].value_counts())
    export_sample_test_flows()
