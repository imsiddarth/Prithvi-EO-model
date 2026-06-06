import sqlite3
from datetime import datetime
import os

DB_PATH = "cog_database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS layers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            layer_name  TEXT,
            cog_path    TEXT,
            wms_url     TEXT,
            bbox        TEXT,
            epsg        TEXT,
            created_at  TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_layer(layer_name, cog_path, wms_url, bbox, epsg):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO layers (layer_name, cog_path, wms_url, bbox, epsg, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (layer_name, cog_path, wms_url, str(bbox), epsg, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_all_layers():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, layer_name, wms_url, bbox, epsg, created_at, cog_path FROM layers ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return rows