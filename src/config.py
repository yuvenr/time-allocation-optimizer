"""Pemuatan konfigurasi prototipe (KNF-07).

Urutan prioritas (yang belakang menimpa yang depan):
1. ``config.yaml``
2. ``config.override.yaml`` (ditulis dari tab Parameter di UI Administrator)
3. Variabel lingkungan ``LLM_PROVIDER`` / ``LLM_MODEL`` (dari ``.env``)
4. Argumen ``overrides`` (dipakai tes)
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
# Hosting (Azure App Service): CRS_CONFIG_OVERRIDE menunjuk ke penyimpanan persisten (/home) agar
# parameter yang diubah dari UI tidak hilang saat kontainer dimulai ulang.
OVERRIDE_PATH = Path(os.getenv("CRS_CONFIG_OVERRIDE") or ROOT / "config.override.yaml")


class PathsCfg(BaseModel):
    raw: str = "data/raw"                   # berkas masukan KRS mentah (impor; tidak pernah di-commit)
    db: str = "data/crs.db"                 # basis data SQLite: data operasional & hasil proses (src/db.py)
    db_url: str = ""                        # postgresql://user:pass@host/db?sslmode=require → backend PostgreSQL
                                            # (hosting cloud). Kosong = SQLite di `db`. Diisi lewat DATABASE_URL (.env /
                                            # Streamlit secrets) — jangan tulis kata sandi di config.yaml.
    db_schema: str = ""                     # skema PostgreSQL opsional (kosong = search_path bawaan, biasanya public)
    ekspor: str = "data/ekspor"             # ekspor CSV untuk JASP / lampiran
    logs: str = "logs"
    kurikulum: str = "data/kurikulum_2024"  # dataset Kurikulum 2024 hasil build (read-only)
    rating: str = "data/rating"             # formulir isian pakar (templat & berkas untuk diimpor)


class KurikulumCfg(BaseModel):
    """Horizon rencana. Angka aturan (145/160 SKS, paket, pita SKS, masa studi) dibaca dari dataset."""

    horizon_semester: int = Field(8, ge=1, le=14)
    perpanjang_melewati_horizon: bool = True   # lanjutkan rencana (≤ AT04) bila belum lulus di horizon


class SkripsiCfg(BaseModel):
    """AD-10 (Pedoman Pendidikan UB Subbab 8.5): syarat mengambil tugas akhir/skripsi."""

    kode: str = "UBU60001"
    min_sks_lulus: int = Field(120, ge=0)
    min_ipk: float = Field(2.0, ge=0, le=4)


class AturanCfg(BaseModel):
    """Tafsiran & parameter aturan domain akademik.

    Nama kunci sengaja sama dengan CLAUDE.md agar mudah dilacak.
    """

    PRASYARAT_E_BERARTI_PERNAH_MENEMPUH: bool = False   # AD-03: prasyarat harus LULUS
    NILAI_LULUS_MINIMUM: str = "D"                      # AD-07: lulus = bukan E (Pedoman UB 8.4)
    SKALA_NILAI: list[str] = Field(default_factory=lambda: ["A", "B+", "B", "C+", "C", "D+", "D", "E"])
    BOBOT_NILAI: dict[str, float] = Field(default_factory=lambda: {
        "A": 4.0, "B+": 3.5, "B": 3.0, "C+": 2.5, "C": 2.0, "D+": 1.5, "D": 1.0, "E": 0.0})
    NILAI_DIBATASI: list[str] = Field(default_factory=lambda: ["D+", "D"])   # AD-07: maks. proporsi SKS
    BATAS_PROPORSI_NILAI_DIBATASI: float = Field(0.10, ge=0, le=1)
    SKRIPSI: SkripsiCfg = Field(default_factory=SkripsiCfg)
    # KF-11/KNF-07: timpa pita SKS 09 per batas bawah IP → [sks_min, sks_max]; None = pakai dataset apa adanya
    BATAS_SKS_IP: dict[float, tuple[int, int]] | None = None
    MODE_PENAWARAN_PILIHAN: Literal["tabel11_ketat", "paritas"] = "paritas"
    PENAWARAN_DARI_REKOMENDASI_09: bool = True

    @field_validator("SKALA_NILAI")
    @classmethod
    def _skala(cls, v: list[str]) -> list[str]:
        v = [x.strip().upper() for x in v]
        if len(v) < 2 or len(set(v)) != len(v):
            raise ValueError("SKALA_NILAI harus berisi ≥ 2 nilai huruf unik (urut dari tertinggi)")
        return v

    @model_validator(mode="after")
    def _ambang_di_skala(self) -> "AturanCfg":
        if self.NILAI_LULUS_MINIMUM.strip().upper() not in self.SKALA_NILAI:
            raise ValueError(f"NILAI_LULUS_MINIMUM {self.NILAI_LULUS_MINIMUM!r} tidak ada di SKALA_NILAI")
        self.BOBOT_NILAI = {k.strip().upper(): float(v) for k, v in self.BOBOT_NILAI.items()}
        if set(self.BOBOT_NILAI) != set(self.SKALA_NILAI):
            raise ValueError("BOBOT_NILAI harus memuat tepat seluruh nilai di SKALA_NILAI")
        self.NILAI_DIBATASI = [x.strip().upper() for x in self.NILAI_DIBATASI]
        if self.BATAS_SKS_IP:
            self.BATAS_SKS_IP = {round(float(k), 2): v for k, v in self.BATAS_SKS_IP.items()}
            for ip, (mn, mx) in self.BATAS_SKS_IP.items():
                if not 0 <= mn <= mx <= 24:
                    raise ValueError(f"BATAS_SKS_IP[{ip}]: harus 0 ≤ sks_min ≤ sks_max ≤ 24")
        if not set(self.NILAI_DIBATASI) <= set(self.SKALA_NILAI):
            raise ValueError("NILAI_DIBATASI harus bagian dari SKALA_NILAI")
        return self

    def bobot(self, nilai: str | None) -> float | None:
        """Bobot angka nilai huruf (untuk IP/IPK); None bila di luar skala."""
        return None if nilai is None else self.BOBOT_NILAI.get(str(nilai).strip().upper())

    def lulus(self, nilai: str | None) -> bool:
        """Nilai huruf ≥ NILAI_LULUS_MINIMUM. Nilai di luar skala dianggap tidak lulus."""
        if nilai is None:
            return False
        n = str(nilai).strip().upper()
        if n not in self.SKALA_NILAI:
            return False
        return self.SKALA_NILAI.index(n) <= self.SKALA_NILAI.index(self.NILAI_LULUS_MINIMUM.strip().upper())


class PreprocessingCfg(BaseModel):
    max_masa_studi: int = 8
    mk_dihapus: Literal["buang", "prefiks_old"] = "buang"   # MK 2020 berjenis 'dihapus' di 11_penyetaraan
    kolom_id: list[str] = Field(default_factory=lambda: ["nim", "id_anon"])
    kolom_identitas: list[str] = Field(default_factory=list)
    proporsi_uji: float = Field(0.2, gt=0, lt=1)


class SensitivitasCfg(BaseModel):
    support: list[float] = Field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    confidence: list[float] = Field(default_factory=lambda: [0.5, 0.65, 0.8])


class ARMCfg(BaseModel):
    min_support: float = Field(0.2, gt=0, le=1)
    min_confidence: float = Field(0.65, gt=0, le=1)
    min_lift: float = 1.0
    lift_ketat: bool = True                # Gambar 2.4: aturan kuat bila lift > min_lift (bukan ≥)
    max_len: int | None = 3
    partisi_semester: bool = True
    buang_konsekuen_wajib: bool = True     # §6: ruang keputusan ARM = MK pilihan
    # utamakan_asli: per semester pakai KRS akademik bila transaksinya ≥ min_transaksi_asli, selain itu
    # riwayat pelengkap | asli | sintetis (satu sumber saja) | gabungan (semua sumber)
    sumber_data: Literal["utamakan_asli", "asli", "sintetis", "gabungan"] = "utamakan_asli"
    min_transaksi_asli: int = Field(20, ge=1)
    sensitivitas: SensitivitasCfg = Field(default_factory=SensitivitasCfg)


class LLMCfg(BaseModel):
    """UC-09 / KNF-02. ``thinking`` = penalaran internal model (Claude: adaptive thinking; OpenAI:
    ``reasoning``); ``effort`` = kedalaman penalaran. Model penalaran terbaru tidak menerima
    ``temperature`` — konsistensi dijaga lewat pengulangan + median."""

    provider: Literal["mock", "anthropic", "openai", "gemini"] = "mock"
    model: str = "mock-bloom-v1"
    temperature: float = 0.0
    thinking: bool = True
    thinking_display: Literal["summarized", "omitted"] = "summarized"
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "high"
    structured_output: bool = True              # JSON schema ketat (provider yang mendukung)
    max_tokens: int = 16000
    anthropic_fallback: bool = True
    max_retry: int = Field(3, ge=0)             # retry karena keluaran tidak valid
    max_retry_jaringan: int = Field(4, ge=0)    # retry SDK untuk galat jaringan/429/5xx (backoff eksponensial)
    timeout_detik: float = Field(300, gt=0)
    konkurensi: int = Field(4, ge=1, le=32)     # permintaan paralel (mock selalu berurutan)
    n_pengulangan: int = Field(3, ge=1)
    mk_per_batch: int = Field(0, ge=0)          # 0 = seluruh MK dalam satu permintaan per pengulangan
    versi_prompt_bloom: str = "v3"


class BebanCfg(BaseModel):
    metode: Literal["jumlah", "rata_tertimbang"] = "jumlah"
    theta: float | None = None                  # angka = ditimpa manual
    theta_sumber: Literal["otomatis", "data", "struktur"] = "otomatis"
    theta_persentil: float = Field(75, gt=0, le=100)
    cl_default: float | None = None             # None = median CL MK lain
    cl_afektif: float | None = Field(1.0, ge=1, le=6)  # CL MK yang seluruh CPMK-nya afektif tanpa penilaian kognitif
    sumber_cl: Literal["heuristik", "llm", "llm_mock", "pakar"] = "heuristik"
    agregasi_cl: Literal["median", "maks", "rata"] = "median"
    kecualikan_kegiatan_transisi: bool = True   # PLP/PKL/PKM/Capstone (16) tidak masuk B_s


class MetrikCfg(BaseModel):
    abaikan_semester_kosong: bool = True


class ValidasiKRSCfg(BaseModel):
    """Validasi rekomendasi semester berjalan terhadap KRS akademik nyata (mahasiswa data uji)."""

    semester: list[int] = Field(default_factory=lambda: [3, 5, 7])
    toleransi_sks: int = Field(0, ge=0)     # |SKS lulus akademik − SKS riwayat tersimpulkan| maksimum


class SimulasiCfg(BaseModel):
    k_potong: list[int] = Field(default_factory=lambda: [2, 3, 4, 5, 6])
    validasi_krs: ValidasiKRSCfg = Field(default_factory=ValidasiKRSCfg)


class GeneratorCfg(BaseModel):
    """src/sintetis.py — riwayat KRS pelengkap, dikalibrasi dari data akademik nyata bila tersedia."""

    n_alumni: int = Field(300, ge=1)
    angkatan_min: int = 2021
    angkatan_max: int = 2022
    ipk_min: float = Field(2.75, ge=0, le=4)    # batas bawah target IPK (riwayat lulus tepat waktu yang wajar)


class PenyesuaianCfg(BaseModel):
    """Iterasi 3: pemindahan MK antarsemester oleh mahasiswa (``src/penyesuaian.py``). Pemindahan yang
    menimbulkan pelanggaran BARU pada ``aturan_keras`` ditolak; pelanggaran baru aturan lain (mis. AD-07
    prioritas MK tertunda) diterima sebagai peringatan untuk DPA."""

    aturan_keras: list[str] = Field(default_factory=lambda: ["AD-02", "AD-03", "AD-04", "AD-06", "AD-08", "AD-09",
                                                             "AD-10", "AD-11", "AD-12", "AD-14", "AD-15"])
    theta_keras: bool = False   # true = pemindahan yang membuat beban semester > θ (mode hybrid) ditolak


class TampilanCfg(BaseModel):
    """Label tampilan UI. ``nama_tema`` = nama tema pilihan berbahasa Indonesia (dokumen [R] hanya memuat nama
    berbahasa Inggris; dataset 06 tidak diubah). Tema tanpa entri memakai nama dataset."""

    nama_tema: dict[str, str] = Field(default_factory=dict)


class AuthCfg(BaseModel):
    """KF-22: peran berproteksi kata sandi (hash PBKDF2 di .env: CRS_HASH_ADMIN, CRS_HASH_DPA)."""

    aktif: bool = True
    peran_terproteksi: list[str] = Field(default_factory=lambda: ["Administrator", "DPA"])
    maks_gagal: int = Field(5, ge=1)        # percobaan gagal per sesi sebelum dikunci


class Config(BaseModel):
    seed: int = 42
    paths: PathsCfg = Field(default_factory=PathsCfg)
    kurikulum: KurikulumCfg = Field(default_factory=KurikulumCfg)
    aturan: AturanCfg = Field(default_factory=AturanCfg)
    preprocessing: PreprocessingCfg = Field(default_factory=PreprocessingCfg)
    arm: ARMCfg = Field(default_factory=ARMCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    beban: BebanCfg = Field(default_factory=BebanCfg)
    metrik: MetrikCfg = Field(default_factory=MetrikCfg)
    simulasi: SimulasiCfg = Field(default_factory=SimulasiCfg)
    generator: GeneratorCfg = Field(default_factory=GeneratorCfg)
    auth: AuthCfg = Field(default_factory=AuthCfg)
    penyesuaian: PenyesuaianCfg = Field(default_factory=PenyesuaianCfg)
    tampilan: TampilanCfg = Field(default_factory=TampilanCfg)

    # ---- helper path -------------------------------------------------
    def _abs(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    @property
    def raw_dir(self) -> Path:
        return self._abs(self.paths.raw)

    @property
    def db_path(self) -> Path:
        return self._abs(self.paths.db)

    @property
    def ekspor_dir(self) -> Path:
        return self._abs(self.paths.ekspor)

    @property
    def logs_dir(self) -> Path:
        return self._abs(self.paths.logs)

    @property
    def kurikulum_dir(self) -> Path:
        return self._abs(self.paths.kurikulum)

    @property
    def rating_dir(self) -> Path:
        return self._abs(self.paths.rating)


def deep_merge(base: dict, extra: dict) -> dict:
    """Gabungkan ``extra`` ke ``base`` secara rekursif (mengembalikan salinan)."""
    hasil = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(hasil.get(k), dict):
            hasil[k] = deep_merge(hasil[k], v)
        else:
            hasil[k] = copy.deepcopy(v)
    return hasil


def _baca_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    overrides: dict[str, Any] | None = None,
    *,
    config_path: Path | None = None,
    use_override_file: bool = True,
    use_env: bool = True,
) -> Config:
    """Muat konfigurasi efektif (KNF-07)."""
    data = _baca_yaml(config_path or CONFIG_PATH)
    if use_override_file:
        data = deep_merge(data, _baca_yaml(OVERRIDE_PATH))
    if use_env:
        load_dotenv(ROOT / ".env", override=False)
        env_llm: dict[str, Any] = {}
        if os.getenv("LLM_PROVIDER"):
            env_llm["provider"] = os.environ["LLM_PROVIDER"].strip().lower()
        if os.getenv("LLM_MODEL"):
            env_llm["model"] = os.environ["LLM_MODEL"].strip()
        if env_llm:
            data = deep_merge(data, {"llm": env_llm})
        # folder data alternatif (dipakai tes UI agar tidak menyentuh data/ aplikasi)
        env_paths = {k: os.environ[f"CRS_{k.upper()}"] for k in ("raw", "db", "ekspor", "logs", "rating", "kurikulum")
                     if os.getenv(f"CRS_{k.upper()}")}
        if os.getenv("DATABASE_URL"):           # backend PostgreSQL (hosting); kosong = SQLite lokal
            env_paths["db_url"] = os.environ["DATABASE_URL"].strip()
        if os.getenv("DATABASE_SCHEMA"):
            env_paths["db_schema"] = os.environ["DATABASE_SCHEMA"].strip()
        if env_paths:
            data = deep_merge(data, {"paths": env_paths})
    if overrides:
        data = deep_merge(data, overrides)
    return Config.model_validate(data)


def read_override() -> dict:
    return _baca_yaml(OVERRIDE_PATH)


def save_override(perubahan: dict[str, Any], ganti: tuple[str, ...] = ()) -> None:
    """Simpan perubahan parameter dari UI ke ``config.override.yaml``. Kunci bertitik di ``ganti``
    (mis. ``"aturan.BATAS_SKS_IP"``) diganti utuh, bukan di-merge (agar entri dapat dihapus)."""
    lama = read_override()
    for kunci in ganti:
        *induk, akhir = kunci.split(".")
        d = lama
        for k in induk:
            d = d.get(k) if isinstance(d, dict) else None
        if isinstance(d, dict):
            d.pop(akhir, None)
    data = deep_merge(lama, perubahan)
    # validasi dulu agar file override tidak pernah berisi nilai rusak
    Config.model_validate(deep_merge(_baca_yaml(CONFIG_PATH), data))
    with open(OVERRIDE_PATH, "w", encoding="utf-8") as f:
        f.write("# Ditulis otomatis oleh UI Administrator. Hapus file ini untuk kembali ke config.yaml.\n")
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def reset_override() -> None:
    if OVERRIDE_PATH.exists():
        OVERRIDE_PATH.unlink()
