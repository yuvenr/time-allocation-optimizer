"""Tes loader dataset Kurikulum 2024 (data/kurikulum_2024) — CLAUDE.md §3, §8."""
from __future__ import annotations

import shutil

import pandas as pd
import pytest

from src.config import load_config
from src.data_loader import FILE_KURIKULUM, baca_csv_kurikulum, cari_siklus, load_kurikulum
from tests.conftest import buat_cfg

CFG = load_config(use_override_file=False, use_env=False)


@pytest.fixture(scope="module")
def kur():
    return load_kurikulum(CFG)


def test_jumlah_baris_sesuai_kamus_data(kur):
    assert (len(kur.mk), len(kur.prasyarat), len(kur.struktur), len(kur.pita_sks)) == (72, 38, 281, 80)
    assert (len(kur.aturan), len(kur.tema), len(kur.penyetaraan), len(kur.jadwal_lapangan)) == (21, 9, 92, 8)


def test_total_sks_jalur(kur):
    """§8: total SKS Jalur A = 145, Jalur B–F = 147 (INK18)."""
    assert kur.total_sks_jalur("A") == 145
    for j in "BCDEF":
        assert kur.total_sks_jalur(j) == 147, j
    per_sem = {s: sum(x.sks for x in kur.slot("A", s)) for s in range(1, 9)}
    assert per_sem == {1: 20, 2: 20, 3: 21, 4: 21, 5: 21, 6: 21, 7: 11, 8: 10}
    # slot per semester konsisten dengan kolom total_sks_semester
    for j in "ABCDEF":
        for s in range(1, 9):
            slot = kur.slot(j, s)
            assert {x.total_sks_semester for x in slot} == {sum(x.sks for x in slot)}, (j, s)


def test_komposisi_sks_sesuai_aturan_akademik(kur):
    """AT02: 118 SKS wajib (agama dihitung sekali) + 27 SKS pilihan (9 tema × 3)."""
    agama = kur.grup("AGAMA")
    assert len(agama) == 5
    wajib = kur.total_sks(kur.wajib - agama) + kur.sks(sorted(agama)[0])
    assert wajib == kur.param_int("AT02", "wajib_min") == 118
    assert sum(t.sks_minimal for t in kur.tema.values()) == kur.param_int("AT02", "pilihan_min") == 27
    assert kur.param_int("AT01", "total_min") == 145 and kur.param_int("AT03", "total_max") == 160
    assert (kur.param_int("AT04", "semester_min"), kur.param_int("AT04", "semester_max")) == (7, 14)


def test_prasyarat_asiklik_dan_fk_valid(kur):
    assert cari_siklus(kur.graf_prasyarat()) is None
    for p in kur.prasyarat:
        assert p.kode_mk in kur and (p.kode_syarat is None or p.kode_syarat in kur)
    for s in kur.struktur:
        assert s.kode_mk in kur or s.jenis_slot != "MK_WAJIB"
    assert all(p.kode_baru in kur for p in kur.penyetaraan if p.kode_baru)


def test_cari_siklus():
    assert cari_siklus({"a": {"b"}, "b": {"c"}}) is None
    assert cari_siklus({"a": {"b"}, "b": {"c"}, "c": {"a"}}) == ["a", "b", "c", "a"]
    assert cari_siklus({"x": {"x"}}) == ["x", "x"]


def test_prasyarat_hanya_ringkasan_tetap_berlaku(kur):
    """INK01–03: tiga prasyarat hanya ada di [R] tetap diberlakukan."""
    pasangan = {(p.kode_mk, p.kode_syarat) for p in kur.prasyarat if p.jenis == "nilai_minimal_mk"}
    assert {("CIE61108", "CIE62104"), ("CIE61110", "CIE62106"), ("CSD60708", "CIE62303")} <= pasangan
    assert all(p.nilai_minimal == "E" for p in kur.prasyarat if p.jenis == "nilai_minimal_mk")


def test_syarat_plp(kur):
    """PLP: korequisit timbal balik + 10 MK Kependidikan (INK30: PLP tidak mensyaratkan dirinya)."""
    kependidikan = {k for k, m in kur.mk.items() if m.kelompok_prodi == "Kependidikan"} - {"CIE61111", "CIE61112"}
    for plp, pasangan in (("CIE61111", "CIE61112"), ("CIE61112", "CIE61111")):
        assert kur.korequisit(plp) == {pasangan}
        assert {p.kode_syarat for p in kur.syarat_kelompok(plp)} == kependidikan
        assert len(kependidikan) == 10
        assert kur.pedoman_eksternal(plp) is not None
    assert {k for k in kur.mk if kur.pedoman_eksternal(k)} == {"CIE61111", "CIE61112", "UBU60002", "COM60061",
                                                               "UBU60001", "UBU60005"}


def test_atribut_semester_mengikuti_keputusan_dataset(kur):
    assert kur.mk["UBU60003"].semester_jalur_a == 1          # INK06: struktur menang atas Deskripsi
    assert kur.mk["CSD60702"].semester_tabel11 == (6,)       # INK05: Tema D semester 6
    assert kur.mk["CIE61212"].semester_paritas_deskripsi == "Ganjil"
    assert kur.mk["CIE62118"].semester_tabel11 == (4,)       # INK08
    assert kur.mk["CIE61402"].semester_tabel11 == (3, 5)
    for k in kur.wajib:
        assert kur.mk[k].semester_jalur_a is not None, k
    for k in kur.pilihan - {"UBU60029", "UBU60030"}:
        assert kur.mk[k].tema and kur.mk[k].semester_tabel11, k


def test_tabel11_konsisten_dengan_07(kur):
    df = baca_csv_kurikulum(CFG, FILE_KURIKULUM["pilihan_tema"])
    assert len(df) == 24
    for r in df.itertuples():
        harap = tuple(int(x) for x in (r.semester_ditawarkan_1, r.semester_ditawarkan_2) if x)
        assert kur.mk[r.kode_mk].semester_tabel11 == harap, r.kode_mk
        assert kur.mk[r.kode_mk].tema == r.tema


def test_pita_sks_inklusif_dan_fallback(kur):
    p = kur.pita("A", 3, 3.00)
    assert (p.jalur, p.sks_min, p.sks_max, p.aksi, p.kode_terdampak) == ("SEMUA", 22, 24, "tambah", ("TEMA_A",))
    assert kur.pita("A", 3, 2.99).sks_max == 21
    assert kur.pita("C", 3, 2.00).sks_max == 18 and kur.pita("C", 3, 1.99).sks_max == 15
    assert kur.pita("A", 3, 1.50).sks_max == 15 and kur.pita("A", 3, 1.49).sks_max == 12
    assert kur.pita("A", 3, 0.0).sks_max == 12 and kur.pita("A", 3, 4.0).sks_max == 24
    # baris jalur spesifik didahulukan
    a6 = kur.pita("A", 6, 3.2)
    assert a6.jalur == "A" and a6.kode_terdampak == ("COM60061",)
    assert kur.pita("B", 7, 2.2).kode_terdampak == ("UBU60005", "MK_LAIN")
    # tanpa baris: semester konversi jalur B–F dan semester > 8
    assert kur.pita("B", 6, 3.2) is None and kur.pita("A", 9, 3.2) is None
    assert kur.pita_umum(3.2) == (22, 24) and kur.pita_umum(2.5) == (19, 21) and kur.pita_umum(1.0) == (0, 12)


def test_catatan_inkonsistensi_pita_tersimpan(kur):
    """INK20/INK21: target_sks yang tidak cocok hitungan tetap disimpan beserta catatannya."""
    a7 = kur.pita("A", 7, 3.5)
    assert a7.target_sks == 21 and "INK20" in (a7.catatan or "")
    s3 = kur.pita("A", 3, 1.0)
    assert s3.target_sks == 12 and "INK21" in (s3.catatan or "")


def test_agama_dan_jalur(kur):
    assert kur.mk_agama("Islam") == "MPK60001" and kur.mk_agama("Budha") == "MPK60005"
    assert kur.mk_agama(None) is None and kur.mk_agama("Lainnya") is None
    assert kur.semester_jalur("MPK60004", "D") == 2
    assert kur.semester_jalur("CIE61111", "F") == 7 and kur.semester_jalur("CIE61402", "A") is None
    assert [s.jenis_slot for s in kur.slot("B", 6)] == ["PILIHAN_KONVERSI"]
    assert kur.slot("B", 6)[0].sks == kur.param_int("AT13", "konversi_max_per_semester") == 20


def test_jadwal_kegiatan_lapangan(kur):
    assert kur.mk_kegiatan_transisi == {"CIE61111", "CIE61112", "UBU60002", "UBU60005", "COM60061"}
    assert {j.semester_krs for j in kur.jadwal("UBU60005")} == {8}
    assert {j.transisi_semester for j in kur.jadwal("CIE61112")} == {"6->7", "7->8"}


# ---- dataset yang dirusak harus ditolak ------------------------------------------------


def _salin_dataset(tmp_path):
    tujuan = tmp_path / "kurikulum"
    shutil.copytree(CFG.kurikulum_dir, tujuan)
    return buat_cfg(tmp_path, paths={"kurikulum": str(tujuan)}), tujuan


def _ubah_csv(path, fn):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    fn(df).to_csv(path, index=False)


def test_tolak_prasyarat_bersiklus(tmp_path):
    cfg, d = _salin_dataset(tmp_path)

    def siklus(df):
        baru = df.iloc[[0]].copy()  # PR001: CIE62209 ← CIE62204; tambahkan CIE62204 ← CIE62209
        baru[["id_prasyarat", "kode_mk", "kode_mk_syarat"]] = ["PRX", "CIE62204", "CIE62209"]
        return pd.concat([df, baru], ignore_index=True)

    _ubah_csv(d / FILE_KURIKULUM["prasyarat"], siklus)
    with pytest.raises(ValueError, match="bersiklus"):
        load_kurikulum(cfg)


def test_tolak_kode_tidak_dikenal(tmp_path):
    cfg, d = _salin_dataset(tmp_path)
    _ubah_csv(d / FILE_KURIKULUM["struktur"], lambda df: df.assign(kode_mk=df.kode_mk.replace("CIE61208", "XXX99999")))
    with pytest.raises(ValueError, match="XXX99999"):
        load_kurikulum(cfg)


def test_tolak_pita_tidak_kontigu(tmp_path):
    cfg, d = _salin_dataset(tmp_path)
    _ubah_csv(d / FILE_KURIKULUM["pita"], lambda df: df.assign(ip_min=df.ip_min.replace("2.50", "2.60")))
    with pytest.raises(ValueError, match="tidak kontigu"):
        load_kurikulum(cfg)
