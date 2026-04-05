from __future__ import annotations

import sqlite3
from pathlib import Path


def initialize_sqlite_registry(db_path: str, schema_sql_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        with open(schema_sql_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
