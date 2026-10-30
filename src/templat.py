"""Templat berkas masukan (iterasi 3): CSV/xlsx berisi nama kolom saja, untuk acuan pengisian data.

Kolom diambil dari konstanta yang dipakai validasi impor (``src.akademik``, ``src.data_loader``,
``src.llm.beban``) sehingga templat tidak pernah menyimpang dari yang diterima sistem. Setiap kolom punya
keterangan (kamus data) yang ditampilkan di UI Administrator dan di ``docs/PANDUAN_ADMINISTRATOR.md``.

``baca_csv_unggahan`` membaca CSV hasil Excel berlocale Indonesia (pemisah ``;``, BOM UTF-8) maupun CSV
berpemisah koma, agar templat yang diisi di Excel dapat langsung diunggah.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import pandas as pd

from src.akademik import KOLOM_IP, KOLOM_KRS
from src.data_loader import KOLOM_KRS_WAJIB
from src.llm.beban import KOLOM_PAKAR, KOLOM_RATING


@dataclass(frozen=True)
class Templat:
    kunci: str
    berkas: str
    judul: str
    dipakai: str                          # tempat unggah di aplikasi
    kolom: tuple[tuple[str, str], ...]    # (nama kolom, keterangan)
    wajib: tuple[str, ...] = ()           # kolom yang wajib terisi (selain itu opsional)

    @property
    def nama_kolom(self) -> list[str]:
        return [k for k, _ in self.kolom]


KET_KRS = {
    "id_anon": "ID mahasiswa yang sudah dianonimkan (mis. M1A2B3C4D). Jangan isi NIM.",
    "angkatan": "Tahun angkatan, mis. 2024.",
    "semester_ke": "Semester mahasiswa saat mengambil MK (1, 2, 3, ...).",
    "periode": "Periode KRS, mis. Ganjil 2026/2027.",
    "kode_mk": "Kode MK. Kode Kurikulum 2020 otomatis disetarakan ke Kurikulum 2024.",
    "nama_mk": "Nama MK (opsional, hanya untuk dibaca).",
    "sks": "SKS MK (bilangan bulat ≥ 1).",
    "kurikulum_kode": "Kode kurikulum dari sistem akademik (opsional).",
}
KET_IP = {
    "id_anon": KET_KRS["id_anon"], "angkatan": KET_KRS["angkatan"], "semester_ke": KET_KRS["semester_ke"],
    "periode": "Periode nilai, mis. Genap 2025/2026.",
    "ip_beban": "IP semester atas seluruh SKS yang diambil (0,00 sampai 4,00; pakai titik desimal).",
    "sks_beban": "SKS yang diambil pada semester itu.",
    "ipk_beban": "IPK kumulatif atas seluruh SKS yang pernah diambil (0 sampai 4).",
    "sksk_beban": "SKS kumulatif yang pernah diambil.",
    "ip_lulus": "IP semester atas SKS yang lulus (0 sampai 4).",
    "sks_lulus": "SKS yang lulus pada semester itu.",
    "ipk_lulus": "IPK kumulatif atas SKS yang lulus (0 sampai 4).",
    "sksk_lulus": "SKS kumulatif yang lulus.",
    "sks_tidak_lulus": "sks_beban dikurangi sks_lulus.",
}
KET_HISTORI = {
    "id_anon": KET_KRS["id_anon"], "angkatan": KET_KRS["angkatan"], "semester_ke": KET_KRS["semester_ke"],
    "kode_mk": KET_KRS["kode_mk"], "sks": KET_KRS["sks"],
    "nilai_huruf": "Nilai akhir: A, B+, B, C+, C, D+, D, atau E.",
    "ip_semester": "IP semester tersebut (0 sampai 4).",
    "kurikulum": "Kurikulum kode MK: 2020 atau 2024.",
    "masa_studi_semester": "Lama studi alumni sampai lulus (semester). Hanya alumni ≤ 8 semester yang ditambang.",
}
KET_PAKAR = {
    "id_cpmk": "ID CPMK dari 02_cpmk.csv, mis. CIE62303-CPMK04 (lihat formulir pakar terisi).",
    "kode_mk": "Kode MK pemilik CPMK (harus cocok dengan id_cpmk).",
    "kode_pakar": "Kode penilai, mis. P1, P2, P3 (tanpa nama).",
    "bloom": "Level Bloom: C1 sampai C6, A untuk CPMK afektif, atau kosong bila belum dinilai.",
    "catatan": "Catatan pakar (opsional).",
}
KET_RATING = {
    "sumber": "llm (atau llm_mock untuk uji coba).",
    "id_cpmk": KET_PAKAR["id_cpmk"], "kode_mk": KET_PAKAR["kode_mk"],
    "penilai": "Nomor pengulangan LLM (1, 2, 3).",
    "model": "Nama model, mis. gpt-6-luna.",
    "versi_prompt": "Versi prompt, mis. bloom-v3.",
    "bloom": "C1 sampai C6 atau A.",
    "ranah": "kognitif, afektif, atau afektif+kognitif.",
    "status": "ok, atau perlu_tinjau_manual.",
    "timestamp": "Waktu penilaian (ISO 8601).",
}
KET_RIWAYAT_NILAI = {
    "kode_mk": "Kode MK yang sudah ditempuh (kode 2020 otomatis disetarakan).",
    "nilai_huruf": KET_HISTORI["nilai_huruf"],
}

TEMPLAT: dict[str, Templat] = {t.kunci: t for t in (
    Templat("krs", "templat_krs_akademik.csv", "KRS akademik (CSV anonim)",
            "Administrator → Data akademik → Impor data akademik → CSV anonim → histori_krs.csv",
            tuple((k, KET_KRS[k]) for k in KOLOM_KRS),
            ("id_anon", "angkatan", "semester_ke", "periode", "kode_mk", "sks")),
    Templat("ip", "templat_ip_semester.csv", "IP/IPK semester (CSV anonim, opsional)",
            "Administrator → Data akademik → Impor data akademik → CSV anonim → ip_semester.csv",
            tuple((k, KET_IP[k]) for k in KOLOM_IP), tuple(KOLOM_IP)),
    Templat("histori", "templat_histori_krs.csv", "Riwayat KRS multisemester alumni (bernilai)",
            "Administrator → Data akademik → Unggah riwayat alumni",
            tuple((k, KET_HISTORI[k]) for k in ["id_anon"] + KOLOM_KRS_WAJIB), tuple(["id_anon"] + KOLOM_KRS_WAJIB)),
    Templat("pakar", "templat_penilaian_pakar.csv", "Penilaian Bloom per CPMK oleh pakar",
            "Administrator → Beban kognitif (LLM) → Impor penilaian pakar",
            tuple((k, KET_PAKAR[k]) for k in KOLOM_PAKAR), ("id_cpmk", "kode_mk", "kode_pakar")),
    Templat("rating_llm", "templat_rating_llm.csv", "Cadangan rating Bloom LLM",
            "Administrator → Mulai di sini → Cadangan & atur ulang → Pulihkan rating LLM",
            tuple((k, KET_RATING.get(k, "Metadata penilaian (opsional).")) for k in KOLOM_RATING),
            ("sumber", "id_cpmk", "kode_mk", "penilai", "model", "versi_prompt", "bloom", "status")),
    Templat("riwayat_nilai", "templat_riwayat_nilai.csv", "Riwayat nilai satu mahasiswa",
            "Mahasiswa → Isi otomatis dari berkas riwayat nilai",
            tuple(KET_RIWAYAT_NILAI.items()), ("kode_mk", "nilai_huruf")),
)}

# xlsx ekspor sistem akademik (dibaca src.akademik.baca_xlsx): nama sheet memuat "Ganjil/Genap <tahun>"
SHEET_XLSX = {
    "KRS Ganjil 2026": ["NIM", "ANGKATAN", "SEMESTER", "KODE MK", "NAMA MK", "SKS", "KURIKULUM"],
    "IP dan IPK Genap 2025": ["NIM", "ANGKATAN", "SEMESTER", "IP BEBAN", "SKS BEBAN", "IPK BEBAN", "SKSK BEBAN",
                              "IP LULUS", "SKS LULUS", "IPK LULUS", "SKSK LULUS"],
}
BERKAS_XLSX = "templat_data_akademik.xlsx"


def csv_kosong(kunci: str) -> bytes:
    """CSV berisi baris nama kolom saja (UTF-8 dengan BOM agar Excel menampilkan huruf dengan benar)."""
    return (",".join(TEMPLAT[kunci].nama_kolom) + "\n").encode("utf-8-sig")


def xlsx_kosong() -> bytes:
    """xlsx dua sheet (KRS, IP/IPK) berisi nama kolom saja, sesuai format ekspor sistem akademik."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for judul, kolom in SHEET_XLSX.items():
            pd.DataFrame(columns=kolom).to_excel(w, sheet_name=judul, index=False)
    return buf.getvalue()


def kamus(kunci: str) -> pd.DataFrame:
    t = TEMPLAT[kunci]
    return pd.DataFrame([{"Kolom": k, "Wajib": "ya" if k in t.wajib else "opsional", "Keterangan": ket}
                         for k, ket in t.kolom])


def baca_csv_unggahan(sumber, **kw) -> pd.DataFrame:
    """Baca CSV unggahan: pemisah koma atau titik koma (Excel berlocale Indonesia), BOM UTF-8 diabaikan,
    nama kolom dirapikan (spasi di tepi dibuang)."""
    data = sumber.getvalue() if hasattr(sumber, "getvalue") else sumber.read() if hasattr(sumber, "read") \
        else open(sumber, "rb").read()
    teks = data.decode("utf-8-sig")
    baris_awal = teks.split("\n", 1)[0]
    try:
        pemisah = csv.Sniffer().sniff(baris_awal, delimiters=",;\t").delimiter
    except csv.Error:
        pemisah = ","
    df = pd.read_csv(io.StringIO(teks), sep=pemisah, **kw)
    df = df.rename(columns=lambda c: str(c).strip())
    if pemisah == ";":   # Excel berlocale Indonesia juga memakai koma desimal: "3,25" → "3.25"
        for c in df.columns:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                angka = df[c].astype(str).str.fullmatch(r"-?\d+,\d+")
                df.loc[angka, c] = df.loc[angka, c].str.replace(",", ".", regex=False)
                if angka.all() and len(df) and kw.get("dtype") is not str:
                    df[c] = pd.to_numeric(df[c])
    return df
