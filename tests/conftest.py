"""Fixture bersama. Semua keluaran ditulis ke folder sementara agar tes tidak
bergantung pada (dan tidak mengubah) isi ``data/`` milik aplikasi.

- Dataset Kurikulum 2024 (``data/kurikulum_2024``) dibaca langsung — read-only.
- ``cfg_mini``: + KRS fiktif buatan tangan (``tests/fixtures/mini``); hasil yang
  diharapkan dihitung manual di docstring tes.
- ``cfg_kosong``: tanpa data KRS (kondisi saat ini: ARM belum aktif).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

import src.config as _config
from src.config import Config, deep_merge

# Isolasi: tes tidak pernah membaca/menulis basis data, log, ekspor, rating, atau raw milik aplikasi.
# Setiap load_config() selama tes memakai folder sementara untuk path itu, kecuali path diberikan eksplisit
# (overrides) atau lewat CRS_<PATH> (tes UI). Dataset kurikulum tetap dibaca dari data/kurikulum_2024.
_TES = Path(tempfile.mkdtemp(prefix="crs-tes-"))
_load_config_asli = _config.load_config
# .env produksi (LLM_PROVIDER=openai + kunci API) tidak boleh terbawa ke tes: load_dotenv(override=False)
# tidak menimpa variabel yang sudah ada, sehingga nilai di bawah ini menang.
os.environ.update({"LLM_PROVIDER": "mock", "LLM_MODEL": "mock-bloom-v1", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "",
                   "GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""})
# DATABASE_URL produksi (PostgreSQL cloud) juga tidak boleh terbawa ke tes. Untuk menjalankan SELURUH tes terhadap
# PostgreSQL, isi CRS_TEST_DATABASE_URL: setiap lokasi basis data tes dipetakan ke skema PostgreSQL tersendiri.
os.environ.update({"DATABASE_URL": "", "DATABASE_SCHEMA": ""})
PG_TES = os.getenv("CRS_TEST_DATABASE_URL", "").strip()


def _load_config_terisolasi(overrides=None, **kw):
    pakai_env = kw.get("use_env", True)
    dasar = {k: str(_TES / v) for k, v in {"db": "crs.db", "logs": "logs", "ekspor": "ekspor", "rating": "rating",
                                           "raw": "raw"}.items()
             if not (pakai_env and os.getenv(f"CRS_{k.upper()}"))}
    # tes tidak pernah memanggil API berbayar: provider mock kecuali tes memintanya eksplisit
    if not (pakai_env and os.getenv("LLM_PROVIDER")):
        dasar_llm = {"provider": "mock", "model": "mock-bloom-v1"}
    else:
        dasar_llm = {}
    cfg = _load_config_asli(deep_merge({"paths": dasar, "llm": dasar_llm}, overrides or {}), **kw)
    if PG_TES and not cfg.paths.db_url:
        from src.db import id_skema_tes
        cfg.paths.db_url = PG_TES
        cfg.paths.db_schema = id_skema_tes(str(cfg.db_path))
    return cfg


_config.load_config = _load_config_terisolasi
load_config = _load_config_terisolasi

MINI = Path(__file__).resolve().parent / "fixtures" / "mini"


def buat_cfg(base: Path, **extra) -> Config:
    ov = {"paths": {"raw": str(base / "raw"), "db": str(base / "crs.db"), "ekspor": str(base / "ekspor"),
                    "logs": str(base / "logs"), "rating": str(base / "rating")}}
    for k, v in extra.items():
        ov[k] = {**ov[k], **v} if k == "paths" else v
    return load_config(ov, use_override_file=False, use_env=False)


def buat_cfg_mini(base: Path, **extra) -> Config:
    """Salin fixture mini ke ``base/raw`` lalu kembalikan config yang menunjuk ke sana."""
    shutil.copytree(MINI / "raw", base / "raw")
    return buat_cfg(base, llm={"provider": "mock", "model": "mock-rubrik-v1"}, **extra)


@pytest.fixture
def cfg_mini(tmp_path) -> Config:
    """Fixture mini baru untuk setiap tes (murah: 54 baris KRS)."""
    return buat_cfg_mini(tmp_path)


@pytest.fixture
def cfg_kosong(tmp_path) -> Config:
    """Tanpa data KRS alumni (``paths.raw`` kosong)."""
    (tmp_path / "raw").mkdir()
    return buat_cfg(tmp_path)
