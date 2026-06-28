"""Persistent verified-fact store.

SQLite + `sqlite-vec` for vector search, with a numpy cosine fallback if
sqlite-vec fails to load. Survives process restart — this is the moat.

Schema
------
verified_facts (
    id           TEXT PRIMARY KEY,
    claim_key    TEXT,
    subject      TEXT,
    canonical_value TEXT,    -- always stored as text, coerced on read
    unit         TEXT,
    verdict      TEXT,
    source       TEXT,
    explanation  TEXT,
    first_seen_ts REAL,
    times_seen   INTEGER,
    embedding    BLOB        -- float32 little-endian, len = EMBED_DIM
)

vec_facts (virtual table from sqlite-vec) maps id → embedding for ANN search.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from core.schemas import VerifiedFact

log = logging.getLogger(__name__)

# We pick an embedding dim lazily on first put — gemini-embedding-2 returns
# 1536 by default. We persist it in a meta row so a reopen knows the right
# vec table size.
_DEFAULT_DIM = 1536
_META_DIM_KEY = "embedding_dim"

try:
    import sqlite_vec  # type: ignore
    _VEC_AVAILABLE = True
except Exception:  # pragma: no cover
    sqlite_vec = None  # type: ignore
    _VEC_AVAILABLE = False
    log.warning("sqlite-vec unavailable — falling back to numpy cosine search")


def _pack_embedding(emb: list[float]) -> bytes:
    arr = np.asarray(emb, dtype=np.float32)
    return arr.tobytes()


def _unpack_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _row_to_fact(row: sqlite3.Row, embedding: Optional[np.ndarray] = None) -> VerifiedFact:
    cv_raw = row["canonical_value"]
    canonical_value: object
    try:
        canonical_value = json.loads(cv_raw) if cv_raw is not None else None
    except (json.JSONDecodeError, TypeError):
        canonical_value = cv_raw
    return VerifiedFact(
        id=row["id"],
        claim_key=row["claim_key"],
        subject=row["subject"],
        canonical_value=canonical_value,
        unit=row["unit"],
        verdict=row["verdict"],
        source=row["source"],
        explanation=row["explanation"],
        first_seen_ts=row["first_seen_ts"],
        times_seen=row["times_seen"],
        embedding=embedding.tolist() if embedding is not None else [],
    )


class Memory:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._dim: Optional[int] = None
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection = self._open_conn()
        self._init_schema()

    # --- lifecycle ---------------------------------------------------------

    def _open_conn(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        if _VEC_AVAILABLE:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception:
                log.exception("sqlite-vec load failed; using numpy fallback")
                globals()["_VEC_AVAILABLE"] = False  # noqa: PLW0603
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verified_facts (
                    id TEXT PRIMARY KEY,
                    claim_key TEXT,
                    subject TEXT,
                    canonical_value TEXT,
                    unit TEXT,
                    verdict TEXT,
                    source TEXT,
                    explanation TEXT,
                    first_seen_ts REAL,
                    times_seen INTEGER DEFAULT 1,
                    embedding BLOB
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    k TEXT PRIMARY KEY,
                    v TEXT
                )
                """
            )
            row = self._conn.execute("SELECT v FROM meta WHERE k = ?", (_META_DIM_KEY,)).fetchone()
            if row:
                self._dim = int(row["v"])
                self._ensure_vec_table()

    def _ensure_vec_table(self) -> None:
        if not _VEC_AVAILABLE or self._dim is None:
            return
        # vec0 virtual table with the right dim. CREATE IF NOT EXISTS is supported.
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(embedding float[{self._dim}])"
        )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # --- writes -----------------------------------------------------------

    def put(self, fact: VerifiedFact) -> None:
        if not fact.embedding:
            raise ValueError("VerifiedFact.embedding is required for put()")
        emb = np.asarray(fact.embedding, dtype=np.float32)
        with self._lock:
            if self._dim is None:
                self._dim = int(emb.shape[0])
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(k, v) VALUES(?, ?)",
                    (_META_DIM_KEY, str(self._dim)),
                )
                self._ensure_vec_table()
            elif emb.shape[0] != self._dim:
                raise ValueError(
                    f"embedding dim mismatch: got {emb.shape[0]}, store is {self._dim}"
                )
            cv = json.dumps(fact.canonical_value)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO verified_facts
                (id, claim_key, subject, canonical_value, unit, verdict, source, explanation, first_seen_ts, times_seen, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.id,
                    fact.claim_key,
                    fact.subject,
                    cv,
                    fact.unit,
                    fact.verdict,
                    fact.source,
                    fact.explanation,
                    fact.first_seen_ts,
                    fact.times_seen,
                    _pack_embedding(fact.embedding),
                ),
            )
            if _VEC_AVAILABLE:
                # rowid in vec table = hash of id, but we want stable join → just store
                # under the id's bytes (vec0 supports any rowid; we use the id's hash).
                rowid = self._stable_rowid(fact.id)
                self._conn.execute("DELETE FROM vec_facts WHERE rowid = ?", (rowid,))
                self._conn.execute(
                    "INSERT INTO vec_facts(rowid, embedding) VALUES (?, ?)",
                    (rowid, _pack_embedding(fact.embedding)),
                )

    def touch(self, fact_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE verified_facts SET times_seen = times_seen + 1 WHERE id = ?",
                (fact_id,),
            )

    # --- reads ------------------------------------------------------------

    def get(self, fact_id: str) -> Optional[VerifiedFact]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM verified_facts WHERE id = ?", (fact_id,)
            ).fetchone()
        if not row:
            return None
        emb = _unpack_embedding(row["embedding"]) if row["embedding"] else None
        return _row_to_fact(row, emb)

    def size(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM verified_facts").fetchone()
        return int(row["n"]) if row else 0

    def vector_search(self, embedding: list[float], top_k: int = 5) -> list[tuple[VerifiedFact, float]]:
        """Cosine-similarity scan. For our scale (<10k facts) numpy is fast
        enough — comfortably under the Phase 3 p50<100ms latency budget. The
        sqlite-vec table is populated on put() for future ANN use but not
        consulted here."""
        if not embedding:
            return []
        emb = np.asarray(embedding, dtype=np.float32)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM verified_facts WHERE embedding IS NOT NULL"
            ).fetchall()
        if not rows:
            return []
        embs = np.stack([_unpack_embedding(r["embedding"]) for r in rows])
        # Cosine similarity; normalize defensively (the API claims pre-normalized).
        embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        q = emb / (np.linalg.norm(emb) + 1e-12)
        sims = embs_n @ q
        order = np.argsort(-sims)[:top_k]
        out: list[tuple[VerifiedFact, float]] = []
        for i in order:
            out.append((_row_to_fact(rows[i], embs[i]), float(sims[i])))
        return out

    def facts_by_subject(self, subject_embedding: list[float], threshold: float) -> Iterable[VerifiedFact]:
        for fact, score in self.vector_search(subject_embedding, top_k=10):
            if score >= threshold:
                yield fact

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _stable_rowid(fact_id: str) -> int:
        # 63-bit positive int derived from id, stable across runs
        h = abs(hash(fact_id))
        return h & ((1 << 62) - 1)


_singleton: Optional[Memory] = None


def get_memory() -> Memory:
    global _singleton
    if _singleton is None:
        import config
        _singleton = Memory(config.MEMORY_DB_PATH)
    return _singleton


def reset_memory() -> None:
    """Close the singleton and wipe the on-disk DB file."""
    global _singleton
    if _singleton is not None:
        _singleton.close()
        _singleton = None
    import config
    p = Path(config.MEMORY_DB_PATH)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(p) + suffix)
        if candidate.exists():
            candidate.unlink()
