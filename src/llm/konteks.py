"""Konteks mata kuliah untuk LLM (CLAUDE.md §5 ``llm/konteks.py``).

LLM dipakai untuk ekstraksi beban kognitif: penilaian Bloom memakai nama,
deskripsi (01), CPMK (02), dan Sub-CPMK (03) — tanpa SKS/semester/kode MK/skor lain agar tidak terjadi
*anchoring*.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import Config
from src.data_loader import Kurikulum, load_konteks, load_kurikulum


@dataclass(frozen=True)
class KonteksMK:
    kode: str
    nama: str
    deskripsi: str
    cpmk: tuple[tuple[str, str], ...]              # (id_cpmk, teks)
    subcpmk: tuple[tuple[str, str], ...]           # (id_subcpmk, teks)

    @property
    def id_cpmk(self) -> list[str]:
        return [i for i, _ in self.cpmk]


@dataclass
class SumberKonteks:
    """Tabel konteks yang dimuat sekali lalu dipakai berulang."""

    kurikulum: Kurikulum
    cpmk: pd.DataFrame
    subcpmk: pd.DataFrame

    @classmethod
    def muat(cls, cfg: Config, kurikulum: Kurikulum | None = None) -> "SumberKonteks":
        return cls(kurikulum or load_kurikulum(cfg), load_konteks(cfg, "cpmk"), load_konteks(cfg, "subcpmk"))

    def mk(self, kode: str) -> KonteksMK:
        m = self.kurikulum.mk[kode]
        cp = self.cpmk[self.cpmk["kode_mk"] == kode]
        sub = self.subcpmk[self.subcpmk["kode_mk"] == kode]
        return KonteksMK(kode=kode, nama=m.nama, deskripsi=m.deskripsi or "",
                         cpmk=tuple(zip(cp["id_cpmk"], cp["teks"])), subcpmk=tuple(zip(sub["id_subcpmk"], sub["teks"])))

    def kode_dengan_cpmk(self) -> list[str]:
        return [k for k in self.kurikulum.kode if (self.cpmk["kode_mk"] == k).any()]
