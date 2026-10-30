"""Pemuatan dataset Kurikulum 2024 (``paths.kurikulum``), data mentah KRS (``data/raw``),
dan hasil proses (basis data SQLite, ``src/db.py``).

- Kurikulum 2024 dibaca apa adanya (read-only) lalu divalidasi integritasnya
  (``validasi_kurikulum``); kode aplikasi tidak pernah mengubah nilai dataset.
- Data mentah divalidasi terhadap skema pydantic agar format yang tidak cocok
  langsung ketahuan.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import BaseModel, ValidationError

from src import db
from src.config import Config
from src.schemas import (KRS, AturanAkademik, JadwalLapangan, MataKuliah, Penyetaraan, PitaSKS, Prasyarat,
                         SlotStruktur, TemaPilihan, Transaksi)

# ---------------------------------------------------------------------
# Validasi generik
# ---------------------------------------------------------------------


def _baca_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return pd.read_csv(path, dtype={"kode": str, "kode_mk": str, "kode_prasyarat": str,
                                    "kode_lama": str, "kode_baru": str})


def validasi_df(df: pd.DataFrame, model: type[BaseModel], nama: str,
                kolom_wajib: Iterable[str] | None = None) -> pd.DataFrame:
    """Pastikan kolom wajib ada dan setiap baris lolos skema pydantic."""
    wajib = list(kolom_wajib or model.model_fields)
    kurang = [k for k in wajib if k not in df.columns]
    if kurang:
        raise ValueError(f"{nama}: kolom wajib tidak ada: {kurang}. Kolom tersedia: {list(df.columns)}")
    kolom = [k for k in model.model_fields if k in df.columns]
    for i, rec in enumerate(df[kolom].to_dict("records")):
        try:
            model.model_validate(rec)
        except ValidationError as e:
            raise ValueError(f"{nama} baris {i + 2}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}") from e
    return df


# ---------------------------------------------------------------------
# Kurikulum 2024 (paths.kurikulum) — sumber tunggal struktur & aturan
# ---------------------------------------------------------------------

FILE_KURIKULUM = {
    "mk": "01_mata_kuliah.csv",
    "prasyarat": "05_prasyarat.csv",
    "tema": "06_tema_pilihan.csv",
    "pilihan_tema": "07_mk_pilihan_tema.csv",
    "struktur": "08_struktur_kurikulum_jalur.csv",
    "pita": "09_rekomendasi_beban_sks.csv",
    "aturan": "10_aturan_akademik.csv",
    "penyetaraan": "11_penyetaraan_mk_2020_2024.csv",
    "jadwal": "16_jadwal_kegiatan_lapangan.csv",
}


def baca_csv_kurikulum(cfg: Config, nama: str) -> pd.DataFrame:
    """Baca satu berkas dataset apa adanya (semua kolom teks, sel kosong = "")."""
    path = cfg.kurikulum_dir / nama
    if not path.exists():
        raise FileNotFoundError(f"Berkas dataset tidak ditemukan: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def _model_baris(df: pd.DataFrame, model: type[BaseModel], nama: str) -> list:
    hasil = []
    for i, rec in enumerate(df.to_dict("records")):
        try:
            hasil.append(model.model_validate(rec))
        except ValidationError as e:
            err = e.errors()[0]
            raise ValueError(f"{nama} baris {i + 2}: {err['loc']} {err['msg']}") from e
    return hasil


def cari_siklus(edges: dict[str, set[str]]) -> list[str] | None:
    """Kembalikan satu siklus (daftar simpul) pada graf ``simpul → syarat``, atau None bila asiklik."""
    putih, abu, hitam = 0, 1, 2
    warna: dict[str, int] = {}
    induk: dict[str, str] = {}

    def kunjungi(u: str) -> list[str] | None:
        warna[u] = abu
        for v in sorted(edges.get(u, ())):
            if warna.get(v, putih) == abu:
                jalur, x = [v, u], u
                while x != v and x in induk:
                    x = induk[x]
                    jalur.append(x)
                return list(reversed(jalur))
            if warna.get(v, putih) == putih:
                induk[v] = u
                s = kunjungi(v)
                if s:
                    return s
        warna[u] = hitam
        return None

    for u in sorted(edges):
        if warna.get(u, putih) == putih:
            s = kunjungi(u)
            if s:
                return s
    return None


@dataclass
class Kurikulum:
    """Kurikulum 2024 PTI UB: master MK + prasyarat + struktur jalur + pita SKS + aturan akademik.

    Semua aturan domain dibaca dari berkas ini (bukan di-hardcode). Objek ini hanya
    berisi data dan helper pencarian; logika aturan ada di ``src/rules.py``. Diperlakukan
    read-only setelah dimuat (hasil pencarian di-cache).
    """

    mk: dict[str, MataKuliah]
    prasyarat: list[Prasyarat] = field(default_factory=list)
    struktur: list[SlotStruktur] = field(default_factory=list)
    pita_sks: list[PitaSKS] = field(default_factory=list)
    aturan: dict[str, AturanAkademik] = field(default_factory=dict)
    tema: dict[str, TemaPilihan] = field(default_factory=dict)
    penyetaraan: list[Penyetaraan] = field(default_factory=list)
    jadwal_lapangan: list[JadwalLapangan] = field(default_factory=list)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def _memo(self, kunci, fn):
        if kunci not in self._cache:
            self._cache[kunci] = fn()
        return self._cache[kunci]

    # ---- dasar ------------------------------------------------------
    def __contains__(self, kode: str) -> bool:
        return kode in self.mk

    @property
    def kode(self) -> list[str]:
        return list(self.mk)

    @property
    def wajib(self) -> frozenset[str]:
        return self._memo("wajib", lambda: frozenset(k for k, m in self.mk.items() if m.sifat == "wajib"))

    @property
    def pilihan(self) -> frozenset[str]:
        return self._memo("pilihan", lambda: frozenset(k for k, m in self.mk.items() if m.sifat == "pilihan"))

    def sks(self, kode: str) -> int:
        return self.mk[kode].sks

    def total_sks(self, kode_list: Iterable[str]) -> int:
        return sum(self.mk[k].sks for k in set(kode_list) if k in self.mk)

    def label(self, kode: str) -> str:
        m = self.mk.get(kode)
        return f"{kode} - {m.nama}" if m else kode

    # ---- prasyarat (05) ----------------------------------------------
    def syarat(self, kode: str, jenis: str | None = None) -> list[Prasyarat]:
        def indeks():
            idx: dict[str, list[Prasyarat]] = {}
            for p in self.prasyarat:
                idx.setdefault(p.kode_mk, []).append(p)
            return idx
        semua = self._memo("syarat", indeks).get(kode, [])
        return [p for p in semua if jenis is None or p.jenis == jenis]

    def prasyarat_nilai(self, kode: str) -> list[Prasyarat]:
        """Syarat ``nilai_minimal_mk`` (mis. 'Nilai MK X ≥ E')."""
        return self.syarat(kode, "nilai_minimal_mk")

    def korequisit(self, kode: str) -> set[str]:
        return {p.kode_syarat for p in self.syarat(kode, "korequisit") if p.kode_syarat}

    def syarat_kelompok(self, kode: str) -> list[Prasyarat]:
        """Syarat ``lulus_kelompok_mk_kependidikan`` (PLP; INK30: PLP tidak mensyaratkan dirinya)."""
        return self.syarat(kode, "lulus_kelompok_mk_kependidikan")

    def pedoman_eksternal(self, kode: str) -> Prasyarat | None:
        s = self.syarat(kode, "pedoman_eksternal")
        return s[0] if s else None

    def graf_prasyarat(self) -> dict[str, set[str]]:
        """Graf urutan (MK → syarat yang harus ditempuh lebih dulu); korequisit tidak termasuk."""
        g: dict[str, set[str]] = {}
        for p in self.prasyarat:
            if p.jenis in ("nilai_minimal_mk", "lulus_kelompok_mk_kependidikan") and p.kode_syarat:
                g.setdefault(p.kode_mk, set()).add(p.kode_syarat)
        return g

    # ---- tema & grup (01, 06) ---------------------------------------
    def mk_tema(self, tema: str) -> frozenset[str]:
        return self._memo(("tema", tema), lambda: frozenset(k for k, m in self.mk.items() if m.tema == tema))

    def grup(self, nama: str) -> frozenset[str]:
        return self._memo(("grup", nama), lambda: frozenset(k for k, m in self.mk.items()
                                                            if m.grup_pilih_satu == nama))

    def mk_agama(self, agama: str | None) -> str | None:
        """Kode MK agama untuk input agama mahasiswa ('Islam' → MPK60001); None bila tidak ada."""
        if not agama:
            return None
        target = f"agama {agama}".strip().lower()
        for k in sorted(self.grup("AGAMA")):
            if self.mk[k].nama.strip().lower() == target:
                return k
        return None

    @property
    def pilihan_agama(self) -> list[str]:
        """Nilai input agama yang dikenali (dari nama MK grup AGAMA)."""
        return [self.mk[k].nama.split(" ", 1)[1] for k in sorted(self.grup("AGAMA"))]

    # ---- struktur jalur (08) ----------------------------------------
    def slot(self, jalur: str, semester: int) -> list[SlotStruktur]:
        def indeks():
            idx: dict[tuple[str, int], list[SlotStruktur]] = {}
            for s in sorted(self.struktur, key=lambda x: x.urutan):
                idx.setdefault((s.jalur, s.semester), []).append(s)
            return idx
        return list(self._memo("slot", indeks).get((jalur, semester), []))

    def semester_jalur(self, kode: str, jalur: str) -> int | None:
        """Semester paket MK wajib dalam struktur jalur (slot agama untuk grup AGAMA)."""
        def cari():
            m = self.mk.get(kode)
            for s in sorted(self.struktur, key=lambda x: (x.semester, x.urutan)):
                if s.jalur == jalur and (s.kode_mk == kode or (s.jenis_slot == "MK_WAJIB_PILIH_SATU"
                                                               and m is not None and m.grup_pilih_satu)):
                    return s.semester
            return None
        return self._memo(("semester_jalur", kode, jalur), cari)

    def total_sks_jalur(self, jalur: str) -> int:
        return sum(s.sks for s in self.struktur if s.jalur == jalur)

    # ---- pita SKS (09) ----------------------------------------------
    def pita(self, jalur: str, semester: int, ip: float) -> PitaSKS | None:
        """Pita untuk (jalur, semester, IP): baris jalur mahasiswa dulu, lalu ``SEMUA``."""
        def indeks():
            idx: dict[tuple[str, int], list[PitaSKS]] = {}
            for p in self.pita_sks:
                idx.setdefault((p.jalur, p.semester), []).append(p)
            return idx
        idx = self._memo("pita", indeks)
        for j in (jalur, "SEMUA"):
            cocok = [p for p in idx.get((j, semester), []) if p.cocok(ip)]
            if cocok:
                return cocok[0]
        return None

    def pita_umum(self, ip: float) -> tuple[int, int] | None:
        """(sks_min, sks_max) berdasarkan AT05 bila pita seragam di seluruh 09 (dipakai untuk semester
        tanpa baris rekomendasi, mis. semester > 8). None bila pita tidak seragam."""
        rentang = {(p.ip_min, p.ip_max, p.sks_min, p.sks_max) for p in self.pita_sks}
        if len({(a, b) for a, b, _, _ in rentang}) != len(rentang):
            return None
        for a, b, lo, hi in rentang:
            if a - 1e-9 <= round(float(ip) + 1e-9, 2) <= b + 1e-9:
                return lo, hi
        return None

    # ---- aturan akademik (10) ---------------------------------------
    def param(self, id_aturan: str, kunci: str) -> str:
        a = self.aturan.get(id_aturan)
        if a is None or kunci not in a.parameter:
            raise KeyError(f"Parameter {id_aturan}.{kunci} tidak ada di 10_aturan_akademik.csv")
        return a.parameter[kunci]

    def param_int(self, id_aturan: str, kunci: str) -> int:
        return int(self.param(id_aturan, kunci))

    # ---- kegiatan lapangan (16) -------------------------------------
    def jadwal(self, kode: str) -> list[JadwalLapangan]:
        return [j for j in self.jadwal_lapangan if kode in j.kode_mk]

    @property
    def mk_kegiatan_transisi(self) -> frozenset[str]:
        return self._memo("transisi", lambda: frozenset(k for j in self.jadwal_lapangan for k in j.kode_mk))


def validasi_kurikulum(kur: Kurikulum) -> None:
    """Pemeriksaan integritas yang menjadi prasyarat constraint engine (galat bila gagal)."""
    kode = set(kur.mk)
    galat: list[str] = []
    for p in kur.prasyarat:
        if p.kode_mk not in kode:
            galat.append(f"05 {p.id}: kode_mk {p.kode_mk} tidak ada di 01")
        if p.kode_syarat and p.kode_syarat not in kode:
            galat.append(f"05 {p.id}: kode_mk_syarat {p.kode_syarat} tidak ada di 01")
        if p.jenis in ("nilai_minimal_mk", "lulus_kelompok_mk_kependidikan", "korequisit") and not p.kode_syarat:
            galat.append(f"05 {p.id}: jenis {p.jenis} tanpa kode_mk_syarat")
    for s in kur.struktur:
        if s.jenis_slot == "MK_WAJIB" and s.kode_mk not in kode:
            galat.append(f"08 Jalur {s.jalur} sem. {s.semester} urutan {s.urutan}: {s.kode_mk} tidak ada di 01")
    for p in kur.penyetaraan:
        if p.kode_baru and p.kode_baru not in kode:
            galat.append(f"11 {p.kode_lama}→{p.kode_baru}: kode_baru tidak ada di 01")
    for j in kur.jadwal_lapangan:
        for k in j.kode_mk:
            if k not in kode:
                galat.append(f"16 {j.kegiatan}: {k} tidak ada di 01")
    for k, m in kur.mk.items():
        if m.tema and m.tema not in kur.tema:
            galat.append(f"01 {k}: tema {m.tema} tidak ada di 06")
        if m.sifat == "wajib" and m.semester_jalur_a is None:
            galat.append(f"01 {k}: MK wajib tanpa semester_rekomendasi_jalur_A")
    for (j, s), grup in _kelompok_pita(kur.pita_sks).items():
        urut = sorted(grup, key=lambda p: p.ip_min)
        if abs(urut[0].ip_min) > 1e-9 or abs(urut[-1].ip_max - 4.0) > 1e-9 or any(
                abs(b.ip_min - round(a.ip_max + 0.01, 2)) > 1e-9 for a, b in zip(urut, urut[1:])):
            galat.append(f"09 jalur {j} sem. {s}: pita IP tidak kontigu 0,00–4,00")
    siklus = cari_siklus(kur.graf_prasyarat())
    if siklus:
        galat.append(f"05: graf prasyarat bersiklus: {' → '.join(siklus)}")
    if galat:
        raise ValueError("Dataset Kurikulum 2024 tidak konsisten:\n- " + "\n- ".join(galat))


def _kelompok_pita(pita: list[PitaSKS]) -> dict[tuple[str, int], list[PitaSKS]]:
    hasil: dict[tuple[str, int], list[PitaSKS]] = {}
    for p in pita:
        hasil.setdefault((p.jalur, p.semester), []).append(p)
    return hasil


def _terapkan_batas_sks(pita: list[PitaSKS], batas: dict[float, tuple[int, int]] | None) -> list[PitaSKS]:
    """KF-11/KNF-07: timpa sks_min/sks_max pita 09 per batas bawah IP (``aturan.BATAS_SKS_IP``)."""
    if not batas:
        return pita
    ip_dataset = {round(p.ip_min, 2) for p in pita}
    asing = sorted(set(batas) - ip_dataset)
    if asing:
        raise ValueError(f"aturan.BATAS_SKS_IP: batas bawah IP {asing} tidak ada di 09 (pilihan: {sorted(ip_dataset)})")
    hasil = []
    for p in pita:
        if round(p.ip_min, 2) in batas:
            mn, mx = batas[round(p.ip_min, 2)]
            catatan = f"sks_min/sks_max {p.sks_min}–{p.sks_max} ditimpa config aturan.BATAS_SKS_IP → {mn}–{mx} (KF-11)"
            p = p.model_copy(update={"sks_min": mn, "sks_max": mx,
                                     "catatan": f"{p.catatan}; {catatan}" if p.catatan else catatan})
        hasil.append(p)
    return hasil


def load_kurikulum(cfg: Config, override: bool | pd.DataFrame = True) -> Kurikulum:
    """Muat & validasi dataset Kurikulum 2024 dari ``paths.kurikulum`` (read-only).

    ``override``: True = terapkan override prasyarat dari basis data (``src/override.py``) dan batas SKS
    dari config; False = dataset murni; DataFrame = override prasyarat calon (untuk validasi sebelum disimpan).
    """
    from src.override import daftar_override_prasyarat, terapkan_prasyarat

    mk_list = _model_baris(baca_csv_kurikulum(cfg, FILE_KURIKULUM["mk"]), MataKuliah, FILE_KURIKULUM["mk"])
    mk = {m.kode: m for m in mk_list}
    if len(mk) != len(mk_list):
        raise ValueError(f"{FILE_KURIKULUM['mk']}: kode_mk duplikat")
    kur = Kurikulum(
        mk=mk,
        prasyarat=_model_baris(baca_csv_kurikulum(cfg, FILE_KURIKULUM["prasyarat"]), Prasyarat, FILE_KURIKULUM["prasyarat"]),
        struktur=_model_baris(baca_csv_kurikulum(cfg, FILE_KURIKULUM["struktur"]), SlotStruktur, FILE_KURIKULUM["struktur"]),
        pita_sks=_terapkan_batas_sks(
            _model_baris(baca_csv_kurikulum(cfg, FILE_KURIKULUM["pita"]), PitaSKS, FILE_KURIKULUM["pita"]),
            cfg.aturan.BATAS_SKS_IP if override is not False else None),
        aturan={a.id: a for a in _model_baris(baca_csv_kurikulum(cfg, FILE_KURIKULUM["aturan"]), AturanAkademik,
                                              FILE_KURIKULUM["aturan"])},
        tema={t.kode: t for t in _model_baris(baca_csv_kurikulum(cfg, FILE_KURIKULUM["tema"]), TemaPilihan, FILE_KURIKULUM["tema"])},
        penyetaraan=_model_baris(baca_csv_kurikulum(cfg, FILE_KURIKULUM["penyetaraan"]), Penyetaraan, FILE_KURIKULUM["penyetaraan"]),
        jadwal_lapangan=_model_baris(baca_csv_kurikulum(cfg, FILE_KURIKULUM["jadwal"]), JadwalLapangan, FILE_KURIKULUM["jadwal"]),
    )
    if override is not False:
        ovr = override if isinstance(override, pd.DataFrame) else daftar_override_prasyarat(cfg)
        if len(ovr):
            kur.prasyarat = terapkan_prasyarat(kur.prasyarat, ovr)
    validasi_kurikulum(kur)
    return kur


# ---------------------------------------------------------------------
# Konteks isi MK (02, 03) — dipakai penilaian Bloom (LLM) & alasan rekomendasi
# ---------------------------------------------------------------------

FILE_KONTEKS = {
    "cpmk": ("02_cpmk.csv", ["id_cpmk", "kode_mk", "nomor", "teks", "bloom_maks_heuristik", "ranah_heuristik",
                             "kko_ambigu"]),
    "subcpmk": ("03_subcpmk.csv", ["id_subcpmk", "kode_mk", "nomor", "teks"]),
}


def load_konteks(cfg: Config, nama: str) -> pd.DataFrame:
    """Muat satu berkas konteks isi MK (apa adanya) dan pastikan kolom yang dipakai ada."""
    berkas, wajib = FILE_KONTEKS[nama]
    df = baca_csv_kurikulum(cfg, berkas)
    kurang = [k for k in wajib if k not in df.columns]
    if kurang:
        raise ValueError(f"{berkas}: kolom wajib tidak ada: {kurang}")
    if "nomor" in df.columns:
        df = df.assign(_n=pd.to_numeric(df["nomor"], errors="coerce")).sort_values(["kode_mk", "_n"]).drop(
            columns="_n").reset_index(drop=True)
    if nama == "cpmk":
        from src.override import terapkan_cpmk

        df = terapkan_cpmk(df, cfg)          # override administrator (4.4.2); teks asli di kolom teks_dataset
    return df


KOLOM_KRS_WAJIB = ["angkatan", "semester_ke", "kode_mk", "sks", "nilai_huruf", "ip_semester",
                   "kurikulum", "masa_studi_semester"]


def load_histori(cfg: Config) -> pd.DataFrame:
    """Muat ``histori_krs.csv``. Kolom ID boleh berupa ``nim`` (data asli) atau ``id_anon``."""
    df = _baca_csv(cfg.raw_dir / "histori_krs.csv")
    if not any(k in df.columns for k in cfg.preprocessing.kolom_id):
        raise ValueError(f"histori_krs.csv: butuh salah satu kolom ID {cfg.preprocessing.kolom_id}")
    df["nilai_huruf"] = df["nilai_huruf"].astype(str).str.strip().str.upper()
    tmp = df.copy()
    if "id_anon" not in tmp.columns:
        tmp["id_anon"] = "x"
    validasi_df(tmp, KRS, "histori_krs.csv", kolom_wajib=KOLOM_KRS_WAJIB)
    return df


# ---------------------------------------------------------------------
# Hasil proses
# ---------------------------------------------------------------------

SPLIT = ("latih", "uji")


def simpan_transaksi(cfg: Config, split: str, transaksi: list[Transaksi]) -> int:
    """Ganti seluruh transaksi ``split`` di basis data."""
    if split not in SPLIT:
        raise ValueError(f"split tidak dikenal: {split}")
    with db.koneksi(cfg) as c:
        c.execute("DELETE FROM transaksi WHERE split = ?", (split,))
        return db.tulis(c, "transaksi", ({"split": split, **t.model_dump()} for t in transaksi))


def ada_transaksi(cfg: Config, split: str | None = None) -> bool:
    return db.baca_ringkasan(cfg, "preprocessing") is not None and (
        split is None or db.jumlah(cfg, "transaksi", "split = ?", (split,)) > 0)


def load_transaksi(cfg: Config, split: str = "latih") -> list[Transaksi]:
    if db.baca_ringkasan(cfg, "preprocessing") is None:
        raise LookupError("transaksi belum ada, jalankan preprocessing dulu")
    df = db.baca(cfg, "SELECT * FROM transaksi WHERE split = ? ORDER BY id_anon, semester_ke", (split,))
    return [Transaksi(id_anon=r.id_anon, angkatan=int(r.angkatan), semester_ke=int(r.semester_ke),
                      ip_semester=None if pd.isna(r.ip_semester) else float(r.ip_semester),
                      items=json.loads(r.items), gagal=json.loads(r.gagal), sumber=r.sumber)
            for r in df.itertuples()]


def transaksi_segar(cfg: Config, split: str = "latih") -> tuple[list[Transaksi], str | None]:
    """Transaksi hanya dipakai bila dibentuk dari data masukan saat ini (sidik cocok).

    Mengembalikan (transaksi, None) atau ([], alasan). Transaksi basi — mis. dibentuk dari KRS
    sintetis lama yang kodenya bukan Kurikulum 2024 — tidak boleh dipakai planner, simulasi,
    maupun isian profil demo, karena kodenya bisa tidak dikenal kurikulum saat ini.
    """
    ringkasan = db.baca_ringkasan(cfg, "preprocessing")
    if ringkasan is None:
        return [], f"transaksi {split} alumni belum ada"
    if ringkasan.get("sidik_data") != sidik_raw(cfg):
        return [], f"transaksi {split} basi (data KRS/kurikulum berubah sejak preprocessing)"
    return load_transaksi(cfg, split), None


def histori_segar(cfg: Config, split: str | None = None) -> tuple[pd.DataFrame, str | None]:
    """Histori KRS alumni ter-normalisasi (dengan nilai huruf) dari basis data, bila segar."""
    _, alasan = transaksi_segar(cfg, split or "latih")
    if alasan:
        return pd.DataFrame(), alasan.replace("transaksi", "histori")
    sql = "SELECT * FROM histori" + (" WHERE split = ?" if split else "") + " ORDER BY id_anon, semester_ke, kode_mk"
    return db.baca(cfg, sql, (split,) if split else ()), None


def load_aturan(cfg: Config) -> pd.DataFrame | None:
    df = db.baca(cfg, "SELECT * FROM aturan ORDER BY lift DESC, confidence DESC, support DESC, anteseden, konsekuen")
    return df if db.baca_ringkasan(cfg, "arm") is not None else None


def load_support_item(cfg: Config) -> pd.DataFrame | None:
    df = db.baca(cfg, "SELECT * FROM support_item ORDER BY semester, support DESC, item")
    return df if db.baca_ringkasan(cfg, "arm") is not None else None


# ---------------------------------------------------------------------
# Status sumber data
# ---------------------------------------------------------------------


def status_data_krs(cfg: Config) -> str:
    """Status riwayat KRS multisemester di ``paths.raw/histori_krs.csv``, diturunkan dari berkas:

    - ``belum_tersedia``: berkas tidak ada;
    - ``sintetis`` / ``fixture``: penanda ``SUMBER_DATA.txt`` diawali kata itu (riwayat pelengkap / tes);
    - ``asli``: selain itu (riwayat alumni dari akademik).
    """
    if not (cfg.raw_dir / "histori_krs.csv").exists():
        return "belum_tersedia"
    penanda = cfg.raw_dir / "SUMBER_DATA.txt"
    if penanda.exists():
        kata = penanda.read_text(encoding="utf-8").strip().split()
        if kata and kata[0].lower() in ("sintetis", "fixture"):
            return kata[0].lower()
    return "asli"


def sumber_data(cfg: Config) -> dict[str, str | None]:
    """Sumber data KRS yang tersedia: KRS akademik (``krs_asli``) dan riwayat multisemester."""
    from src.akademik import ada

    riwayat = status_data_krs(cfg)
    return {"krs_akademik": "asli" if ada(cfg) else None,
            "riwayat": None if riwayat == "belum_tersedia" else riwayat}


def label_sumber_data(cfg: Config) -> str:
    """Ringkas: ``belum_tersedia`` | ``asli`` | ``asli+sintetis`` | ``sintetis`` | ``fixture`` | …"""
    s = sumber_data(cfg)
    bagian = list(dict.fromkeys(x for x in (s["krs_akademik"], s["riwayat"]) if x))
    return "+".join(bagian) or "belum_tersedia"


def sidik_raw(cfg: Config) -> str:
    """Sidik jari data masukan (riwayat KRS + data akademik + dataset kurikulum) — untuk mendeteksi hasil
    proses basi. Akhir baris dinormalkan (CRLF = LF): isi yang sama tetap bersidik sama di Windows (git autocrlf)
    dan di server Linux (Streamlit Community Cloud), sehingga hasil proses yang dipindah ke cloud tidak basi."""
    from src.akademik import sidik

    h = hashlib.sha256()
    files = [cfg.raw_dir / "histori_krs.csv"] + [cfg.kurikulum_dir / f for f in FILE_KURIKULUM.values()]
    for f in files:
        if f.exists():
            h.update(f.name.encode())
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    h.update(sidik(cfg).encode())
    return h.hexdigest()[:16]
