import sqlite3
import json
from typing import Any, Optional, Iterable

class DiskKVStore:
    """
    A persistent Key-Value store backed by SQLite.
    Used to replace in-memory dicts for out-of-core graph scaling (OOM prevention).
    """
    def __init__(self, db_path: str, table_name: str = "kvstore"):
        self.db_path = db_path
        self.table_name = table_name
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute(
                f'''CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )'''
            )
            # Optimize for high-speed writes and reads
            self.conn.execute("PRAGMA synchronous = OFF")
            self.conn.execute("PRAGMA journal_mode = WAL")

    def set(self, key: str, value: Any):
        val_str = json.dumps(value)
        with self.conn:
            self.conn.execute(
                f"INSERT OR REPLACE INTO {self.table_name} (key, value) VALUES (?, ?)",
                (key, val_str)
            )

    def get(self, key: str, default: Any = None) -> Any:
        cursor = self.conn.execute(f"SELECT value FROM {self.table_name} WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return default

    def pop(self, key: str, default: Any = None) -> Any:
        val = self.get(key, default)
        with self.conn:
            self.conn.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
        return val

    def keys(self) -> Iterable[str]:
        cursor = self.conn.execute(f"SELECT key FROM {self.table_name}")
        for row in cursor:
            yield row[0]

    def items(self) -> Iterable:
        cursor = self.conn.execute(f"SELECT key, value FROM {self.table_name}")
        for row in cursor:
            yield row[0], json.loads(row[1])

    def values(self) -> Iterable[Any]:
        cursor = self.conn.execute(f"SELECT value FROM {self.table_name}")
        for row in cursor:
            yield json.loads(row[0])

    def __len__(self) -> int:
        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return cursor.fetchone()[0]

    def __iter__(self) -> Iterable[str]:
        return self.keys()

    def vacuum(self) -> None:
        """Compacta o banco SQLite."""
        self.conn.execute("VACUUM")

    def clear(self):
        with self.conn:
            self.conn.execute(f"DELETE FROM {self.table_name}")
        self.vacuum()

    def close(self):
        self.conn.close()

    def __contains__(self, key: str) -> bool:
        cursor = self.conn.execute(f"SELECT 1 FROM {self.table_name} WHERE key = ?", (key,))
        return cursor.fetchone() is not None

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key: str, value: Any):
        self.set(key, value)
