import os
import sqlite3
import json
import threading
from typing import Any, Iterator, Dict, List, Tuple

class DiskKVStore:
    """
    A thread-safe persistent Key-Value store backed by SQLite with WAL mode.
    Used to replace in-memory dicts for out-of-core graph scaling (OOM prevention).
    """
    def __init__(self, db_path: str, table_name: str = "kvstore"):
        self.db_path = db_path
        self.table_name = "".join(c for c in table_name if c.isalnum() or c == "_")
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self._create_table()

    def _create_table(self):
        with self._lock:
            with self.conn:
                self.conn.execute("PRAGMA journal_mode = WAL")
                self.conn.execute("PRAGMA synchronous = NORMAL")
                self.conn.execute("PRAGMA busy_timeout = 10000")
                self.conn.execute(
                    f'''CREATE TABLE IF NOT EXISTS "{self.table_name}" (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )'''
                )

    def set(self, key: str, value: Any) -> None:
        val_str = json.dumps(value, ensure_ascii=False)
        with self._lock:
            with self.conn:
                self.conn.execute(
                    f'INSERT OR REPLACE INTO "{self.table_name}" (key, value) VALUES (?, ?)',
                    (key, val_str)
                )

    def set_many(self, mapping: Dict[str, Any]) -> None:
        """High-throughput atomic batch insert using a single transaction."""
        if not mapping:
            return
        items = [(k, json.dumps(v, ensure_ascii=False)) for k, v in mapping.items()]
        with self._lock:
            with self.conn:
                self.conn.executemany(
                    f'INSERT OR REPLACE INTO "{self.table_name}" (key, value) VALUES (?, ?)',
                    items
                )

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            cursor = self.conn.execute(
                f'SELECT value FROM "{self.table_name}" WHERE key = ?',
                (key,)
            )
            row = cursor.fetchone()
            if row is not None:
                return json.loads(row[0])
            return default

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            cursor = self.conn.execute(
                f'SELECT value FROM "{self.table_name}" WHERE key = ?',
                (key,)
            )
            row = cursor.fetchone()
            if row is None:
                return default
            val = json.loads(row[0])
            with self.conn:
                self.conn.execute(
                    f'DELETE FROM "{self.table_name}" WHERE key = ?',
                    (key,)
                )
            return val

    def iter_keys(self, batch_size: int = 2000) -> Iterator[str]:
        """Iterador com streaming paginado para evitar picos de memória RAM."""
        offset = 0
        while True:
            with self._lock:
                cursor = self.conn.execute(
                    f'SELECT key FROM "{self.table_name}" LIMIT ? OFFSET ?',
                    (batch_size, offset)
                )
                rows = cursor.fetchall()
            if not rows:
                break
            for row in rows:
                yield row[0]
            if len(rows) < batch_size:
                break
            offset += len(rows)

    def iter_items(self, batch_size: int = 2000) -> Iterator[Tuple[str, Any]]:
        """Iterador com streaming paginado de pares (chave, valor) decodificados."""
        offset = 0
        while True:
            with self._lock:
                cursor = self.conn.execute(
                    f'SELECT key, value FROM "{self.table_name}" LIMIT ? OFFSET ?',
                    (batch_size, offset)
                )
                rows = cursor.fetchall()
            if not rows:
                break
            for row in rows:
                yield (row[0], json.loads(row[1]))
            if len(rows) < batch_size:
                break
            offset += len(rows)

    def iter_values(self, batch_size: int = 2000) -> Iterator[Any]:
        """Iterador com streaming paginado de valores decodificados."""
        offset = 0
        while True:
            with self._lock:
                cursor = self.conn.execute(
                    f'SELECT value FROM "{self.table_name}" LIMIT ? OFFSET ?',
                    (batch_size, offset)
                )
                rows = cursor.fetchall()
            if not rows:
                break
            for row in rows:
                yield json.loads(row[0])
            if len(rows) < batch_size:
                break
            offset += len(rows)

    def keys(self) -> List[str]:
        return list(self.iter_keys())

    def items(self) -> List[Tuple[str, Any]]:
        return list(self.iter_items())

    def values(self) -> List[Any]:
        return list(self.iter_values())

    def __len__(self) -> int:
        with self._lock:
            cursor = self.conn.execute(f'SELECT COUNT(*) FROM "{self.table_name}"')
            return cursor.fetchone()[0]

    def __iter__(self) -> Iterator[str]:
        return self.iter_keys()

    def vacuum(self) -> None:
        """Compacta o banco SQLite."""
        with self._lock:
            self.conn.execute("VACUUM")

    def clear(self) -> None:
        with self._lock:
            with self.conn:
                self.conn.execute(f'DELETE FROM "{self.table_name}"')
            self.vacuum()

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.commit()
            except Exception:
                pass
            self.conn.close()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            cursor = self.conn.execute(
                f'SELECT 1 FROM "{self.table_name}" WHERE key = ?',
                (key,)
            )
            return cursor.fetchone() is not None

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            cursor = self.conn.execute(
                f'SELECT value FROM "{self.table_name}" WHERE key = ?',
                (key,)
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(key)
            return json.loads(row[0])

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def backup_to(self, dest_path: str) -> None:
        """Create an atomic, streaming SQLite backup to dest_path without in-memory deserialization."""
        dest_abs = os.path.abspath(dest_path)
        src_abs = os.path.abspath(self.db_path)
        if dest_abs == src_abs:
            return
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        with self._lock:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:
                pass
            dest_conn = sqlite3.connect(dest_path)
            try:
                self.conn.backup(dest_conn)
            finally:
                dest_conn.close()

    def restore_from(self, src_path: str) -> None:
        """Restore database contents from another SQLite database file using streaming backup."""
        if not os.path.exists(src_path):
            return
        src_abs = os.path.abspath(src_path)
        dest_abs = os.path.abspath(self.db_path)
        if src_abs == dest_abs:
            return
        with self._lock:
            src_conn = sqlite3.connect(src_path)
            try:
                src_conn.backup(self.conn)
            finally:
                src_conn.close()

