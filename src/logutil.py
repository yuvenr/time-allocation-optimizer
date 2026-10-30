"""Pencatatan proses (KNF-09).

Setiap proses (preprocessing, penambangan, ekstraksi, simulasi) dicatat sebagai satu baris di
tabel ``log_proses`` (SQLite) beserta parameter, ringkasan hasil, waktu mulai/selesai, durasi,
dan status data KRS. Ringkasan teks juga ditulis ke ``logs/app.log``.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from src import db
from src.config import Config
from src.data_loader import label_sumber_data


def buat_run_id(proses: str) -> str:
    return f"{proses}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def get_logger(cfg: Config, nama: str = "crs") -> logging.Logger:
    logger = logging.getLogger(nama)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    target = str(cfg.logs_dir / "app.log")
    if not any(getattr(h, "baseFilename", None) == str(Path(target).resolve()) for h in logger.handlers):
        h = logging.FileHandler(target, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def teruskan_log_sdk(cfg: Config) -> None:
    """Teruskan log INFO SDK penyedia LLM & pustaka HTTP-nya ke ``logs/app.log`` (pelengkap). Jejak utama setiap
    percobaan HTTP — termasuk 429/5xx/putus yang di-retry SDK — dicatat sendiri oleh
    ``src.llm.client.transport_tercatat`` (logger ``crs.llm.http``) karena SDK mencatat retry di level DEBUG
    dan memakai ``httpx2``, bukan ``httpx``."""
    handler = get_logger(cfg).handlers
    for nama in ("openai", "anthropic", "httpx", "httpx2", "google_genai"):
        lg = logging.getLogger(nama)
        lg.setLevel(logging.INFO)
        for h in handler:
            if h not in lg.handlers:
                lg.addHandler(h)


def _json_default(o: Any):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, (datetime, Path)):
        return str(o)
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    return str(o)


class CatatanProses:
    def __init__(self, proses: str, parameter: dict[str, Any], run_id: str):
        self.proses = proses
        self.parameter = parameter
        self.run_id = run_id
        self.ringkasan: dict[str, Any] = {}


@contextmanager
def catat_proses(cfg: Config, proses: str, parameter: dict[str, Any] | None = None) -> Iterator[CatatanProses]:
    """Context manager: catat parameter, durasi, status, dan ringkasan (KNF-09)."""
    catatan = CatatanProses(proses, dict(parameter or {}), buat_run_id(proses))
    mulai = datetime.now()
    t0 = time.perf_counter()
    status = "ok"
    galat = None
    try:
        yield catatan
    except Exception as e:  # dicatat lalu dilempar ulang
        status, galat = "error", repr(e)
        raise
    finally:
        entri = {
            "run_id": catatan.run_id,
            "proses": proses,
            "mulai": mulai.isoformat(timespec="seconds"),
            "selesai": datetime.now().isoformat(timespec="seconds"),
            "durasi_detik": round(time.perf_counter() - t0, 3),
            "status": status,
            "seed": cfg.seed,
            "data_krs": label_sumber_data(cfg),
            "parameter": catatan.parameter,
            "ringkasan": catatan.ringkasan,
        }
        with db.koneksi(cfg) as c:
            db.tulis(c, "log_proses", [{**entri, "galat": galat,
                                        "parameter": json.dumps(entri["parameter"], ensure_ascii=False,
                                                                default=_json_default),
                                        "ringkasan": json.dumps(entri["ringkasan"], ensure_ascii=False,
                                                                default=_json_default)}])
        get_logger(cfg).info("%s %s (%.2fs) %s", proses, status, entri["durasi_detik"], catatan.run_id)


def baca_log(cfg: Config, proses: str | None = None, n: int = 50) -> list[dict]:
    """``n`` entri log terakhir (urut kronologis), opsional hanya satu jenis proses."""
    sql = "SELECT * FROM log_proses" + (" WHERE proses = ?" if proses else "") + " ORDER BY mulai DESC, rowid DESC LIMIT ?"
    df = db.baca(cfg, sql, ((proses,) if proses else ()) + (n,))
    baris = []
    for r in df.to_dict("records")[::-1]:
        r["parameter"] = json.loads(r["parameter"] or "{}")
        r["ringkasan"] = json.loads(r["ringkasan"] or "{}")
        if not r.get("galat"):
            r.pop("galat", None)
        baris.append(r)
    return baris
