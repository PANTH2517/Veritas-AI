import os

import sqlite3

import uuid

import json

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_DIR = os.path.join(BASE_DIR, "data")
if os.environ.get("VERCEL") or not os.access(DB_DIR, os.W_OK):
    DB_PATH = "/tmp/deepfake_signatures.db"
else:
    DB_PATH = os.path.join(DB_DIR, "deepfake_signatures.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_db_connection():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    return conn

def init_db():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_signatures (
            id TEXT PRIMARY KEY,
            boundary_discontinuity REAL,
            texture_anomaly REAL,
            channel_asymmetry REAL,
            feature_contrast_anomaly REAL,
            local_blur_factor REAL,
            frequency_dct_spikes REAL,
            bilateral_asymmetry REAL,
            edge_density_deviation REAL,
            label INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_signatures (
            id TEXT PRIMARY KEY,
            high_frequency_energy_anomaly REAL,
            phase_incoherence REAL,
            spectral_flux_deviation REAL,
            spectral_centroid_deviation REAL,
            spectral_rolloff_deviation REAL,
            spectral_bandwidth_anomaly REAL,
            label INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id TEXT PRIMARY KEY,
            filename TEXT,
            media_type TEXT,
            risk_score REAL,
            risk_grade TEXT,
            features TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM face_signatures")

    face_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM voice_signatures")

    voice_count = cursor.fetchone()[0]

    if face_count == 0:

        seed_face_signatures(conn)

    if voice_count == 0:

        seed_voice_signatures(conn)

    conn.close()

def seed_face_signatures(conn):

    print("Seeding face_signatures table in SQLite...")

    cursor = conn.cursor()

    np.random.seed(42)

    n_samples = 1000

    real_features = np.zeros((n_samples, 8))

    real_features[:, 0] = np.random.normal(0.12, 0.05, n_samples)           

    real_features[:, 1] = np.random.normal(0.14, 0.06, n_samples)          

    real_features[:, 2] = np.random.normal(0.10, 0.04, n_samples)          

    real_features[:, 3] = np.random.normal(0.15, 0.05, n_samples)           

    real_features[:, 4] = np.random.normal(0.11, 0.04, n_samples)       

    real_features[:, 5] = np.random.normal(0.08, 0.03, n_samples)      

    real_features[:, 6] = np.random.normal(0.12, 0.05, n_samples)           

    real_features[:, 7] = np.random.normal(0.10, 0.04, n_samples)       

    fake_features = np.zeros((n_samples, 8))

    fake_features[:, 0] = np.random.normal(0.68, 0.12, n_samples)

    fake_features[:, 1] = np.random.normal(0.62, 0.14, n_samples)

    fake_features[:, 2] = np.random.normal(0.55, 0.10, n_samples)

    fake_features[:, 3] = np.random.normal(0.48, 0.12, n_samples)

    fake_features[:, 4] = np.random.normal(0.52, 0.11, n_samples)

    fake_features[:, 5] = np.random.normal(0.65, 0.13, n_samples)

    fake_features[:, 6] = np.random.normal(0.50, 0.12, n_samples)

    fake_features[:, 7] = np.random.normal(0.48, 0.10, n_samples)

    real_features = np.clip(real_features, 0.0, 1.0)

    fake_features = np.clip(fake_features, 0.0, 1.0)

    for i in range(n_samples):

        row = real_features[i]

        cursor.execute("""
            INSERT INTO face_signatures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (str(uuid.uuid4()), *[float(v) for v in row]))

    for i in range(n_samples):

        row = fake_features[i]

        cursor.execute("""
            INSERT INTO face_signatures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (str(uuid.uuid4()), *[float(v) for v in row]))

    conn.commit()

    print("Face signatures seeded successfully.")

def seed_voice_signatures(conn):

    print("Seeding voice_signatures table in SQLite...")

    cursor = conn.cursor()

    np.random.seed(43)

    n_samples = 1000

    real_features = np.zeros((n_samples, 6))

    real_features[:, 0] = np.random.normal(0.12, 0.04, n_samples)     

    real_features[:, 1] = np.random.normal(0.08, 0.03, n_samples)        

    real_features[:, 2] = np.random.normal(0.15, 0.05, n_samples)       

    real_features[:, 3] = np.random.normal(0.14, 0.05, n_samples)           

    real_features[:, 4] = np.random.normal(0.16, 0.06, n_samples)          

    real_features[:, 5] = np.random.normal(0.13, 0.05, n_samples)            

    fake_features = np.zeros((n_samples, 6))

    fake_features[:, 0] = np.random.normal(0.72, 0.10, n_samples)

    fake_features[:, 1] = np.random.normal(0.68, 0.12, n_samples)

    fake_features[:, 2] = np.random.normal(0.58, 0.12, n_samples)

    fake_features[:, 3] = np.random.normal(0.50, 0.11, n_samples)

    fake_features[:, 4] = np.random.normal(0.62, 0.13, n_samples)

    fake_features[:, 5] = np.random.normal(0.55, 0.11, n_samples)

    real_features = np.clip(real_features, 0.0, 1.0)

    fake_features = np.clip(fake_features, 0.0, 1.0)

    for i in range(n_samples):

        row = real_features[i]

        cursor.execute("""
            INSERT INTO voice_signatures VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        """, (str(uuid.uuid4()), *[float(v) for v in row]))

    for i in range(n_samples):

        row = fake_features[i]

        cursor.execute("""
            INSERT INTO voice_signatures VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        """, (str(uuid.uuid4()), *[float(v) for v in row]))

    conn.commit()

    print("Voice signatures seeded successfully.")

def load_voice_signatures_from_db():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='voice_signatures'")

    if not cursor.fetchone():

        conn.close()

        return np.array([]), np.array([])

    cursor.execute("SELECT * FROM voice_signatures")

    rows = cursor.fetchall()

    conn.close()

    features = []

    labels = []

    for row in rows:

        features.append([

            row['high_frequency_energy_anomaly'],

            row['phase_incoherence'],

            row['spectral_flux_deviation'],

            row['spectral_centroid_deviation'],

            row['spectral_rolloff_deviation'],

            row['spectral_bandwidth_anomaly']

        ])

        labels.append(row['label'])

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)

def save_scan_to_history(scan_id, filename, media_type, risk_score, risk_grade, result_dict):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO scan_history (id, filename, media_type, risk_score, risk_grade, features)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (scan_id, filename, media_type, risk_score, risk_grade, json.dumps(result_dict)))

    conn.commit()

    conn.close()

def get_scan_from_history(scan_id):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("SELECT features FROM scan_history WHERE id = ?", (scan_id,))

    row = cursor.fetchone()

    conn.close()

    if row:

        return json.loads(row['features'])

    return None

def get_recent_scans(limit=30):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, filename, media_type, risk_score, risk_grade, timestamp
        FROM scan_history
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]

def delete_scan_from_history(scan_id):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("DELETE FROM scan_history WHERE id = ?", (scan_id,))

    deleted = cursor.rowcount

    conn.commit()

    conn.close()

    return deleted > 0

def clear_all_history():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("DELETE FROM scan_history")

    conn.commit()

    conn.close()

