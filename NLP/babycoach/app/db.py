from __future__ import annotations

import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "babycoach.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS baby_profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                birth_date TEXT,
                age_months INTEGER,
                weight_kg REAL,
                allergies TEXT,
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS baby_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                baby_id INTEGER NOT NULL,
                wisdom TEXT,
                happiness TEXT,
                growth_direction TEXT,
                FOREIGN KEY (baby_id) REFERENCES baby_profile(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                baby_id INTEGER,
                type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (baby_id) REFERENCES baby_profile(id) ON DELETE SET NULL
            )
            """
        )

        # Backward-compatible migration for previously created DB files.
        profile_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(baby_profile)").fetchall()
        }
        if "baby_photo_name" not in profile_cols:
            conn.execute("ALTER TABLE baby_profile ADD COLUMN baby_photo_name TEXT")
        if "baby_photo_url" not in profile_cols:
            conn.execute("ALTER TABLE baby_profile ADD COLUMN baby_photo_url TEXT")

        conn.commit()

