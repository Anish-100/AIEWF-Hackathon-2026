
"""Persistent fact memory with vector search (SQLite + numpy cosine)."""
import sqlite3
import time
from pathlib import Path

import numpy as np

from .schemas import VerifiedFact

DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                subject   TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value     TEXT NOT NULL,
                source    TEXT,
                ts        REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_subject ON facts(subject);
            CREATE TABLE IF NOT EXISTS embeddings (
                fact_id INTEGER PRIMARY KEY REFERENCES facts(id),
                vector  BLOB NOT NULL
            );
        """)


def put(subject: str, predicate: str, value: str, source: str,
        embedding: np.ndarray) -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO facts(subject, predicate, value, source, ts) VALUES(?,?,?,?,?)",
            (subject, predicate, value, source, time.time()),
        )
        fact_id = cur.lastrowid
        con.execute(
            "INSERT OR REPLACE INTO embeddings(fact_id, vector) VALUES(?,?)",
            (fact_id, embedding.astype(np.float32).tobytes()),
        )
        return fact_id


def get(subject: str, predicate: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT value, source, ts FROM facts "
            "WHERE subject=? AND predicate=? ORDER BY ts DESC LIMIT 1",
            (subject, predicate),
        ).fetchone()
    if row:
        return {"value": row[0], "source": row[1], "ts": row[2]}
    return None


def facts_by_subject(subject: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT predicate, value, source, ts FROM facts WHERE subject=?",
            (subject,),
        ).fetchall()
    return [{"predicate": r[0], "value": r[1], "source": r[2], "ts": r[3]} for r in rows]


def vector_search(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    q = query_embedding.astype(np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    with _conn() as con:
        rows = con.execute(
            "SELECT f.id, f.subject, f.predicate, f.value, e.vector "
            "FROM facts f JOIN embeddings e ON f.id = e.fact_id"
        ).fetchall()
    results = []
    for fid, subj, pred, val, blob in rows:
        v = np.frombuffer(blob, dtype=np.float32).copy()
        v_norm = v / (np.linalg.norm(v) + 1e-9)
        score = float(np.dot(q_norm, v_norm))
        results.append({
            "id": fid, "subject": subj, "predicate": pred,
            "value": val, "score": score, "_embedding": v,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def touch(fact_id: int) -> None:
    with _conn() as con:
        con.execute("UPDATE facts SET ts=? WHERE id=?", (time.time(), fact_id))


def clear() -> None:
    with _conn() as con:
        con.execute("DELETE FROM embeddings")
        con.execute("DELETE FROM facts")
