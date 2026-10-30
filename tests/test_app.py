"""Tes asap UI Streamlit (tanpa browser) memakai AppTest pada folder data sementara.

Dua kondisi: tanpa data KRS dan dengan riwayat KRS fixture buatan tangan. Kurikulum = dataset asli
``data/kurikulum_2024`` (hanya dibaca); provider LLM = mock; peran Administrator/DPA memakai kata sandi
uji (hash di variabel lingkungan, KF-22).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import db
from src.auth import buat_hash
from src.config import load_config
from tests.conftest import MINI

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
SANDI = "sandi-uji-123"


def _app(tmp_path, monkeypatch, dengan_krs: bool, siapkan=None) -> AppTest:
    raw = tmp_path / "raw"
    if dengan_krs:
        shutil.copytree(MINI / "raw", raw)
    else:
        raw.mkdir()
    for k in ("raw", "ekspor", "logs", "rating"):
        monkeypatch.setenv(f"CRS_{k.upper()}", str(tmp_path / k))
    monkeypatch.setenv("CRS_DB", str(tmp_path / "crs.db"))
    if siapkan:
        siapkan(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_MODEL", "mock-bloom-v1")
    for var in ("CRS_HASH_ADMIN", "CRS_HASH_DPA"):
        monkeypatch.setenv(var, buat_hash(SANDI, iterasi=1000))
    monkeypatch.setattr("src.config.OVERRIDE_PATH", tmp_path / "config.override.yaml")
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.fixture
def app(tmp_path, monkeypatch):
    return _app(tmp_path, monkeypatch, dengan_krs=False)


@pytest.fixture
def app_krs(tmp_path, monkeypatch):
    return _app(tmp_path, monkeypatch, dengan_krs=True)


def _pilih_peran(at, peran, sandi: str | None = SANDI):
    """Pilih peran; Administrator/DPA masuk dengan kata sandi (sekali per sesi)."""
    at.sidebar.radio[0].set_value(peran).run()
    assert not at.exception, at.exception
    if sandi is not None and any(b.label == "Masuk" for b in at.button):
        at.text_input[0].set_value(sandi)
        _tombol(at, "Masuk").click().run()
        assert not at.exception, at.exception


def _tombol(at, label):
    return next(b for b in at.button if b.label == label)


def _buka_tab(at, label):
    """Tab Administrator dirender hanya saat dibuka (st.tabs on_change="rerun"); buka seperti klik pengguna."""
    at.session_state["tab_admin"] = label
    at.run()
    assert not at.exception, at.exception


def _klik_di_tab(at, tab, label):
    """AppTest tidak menyimpan tab aktif antar-run (di peramban, frontend mengirimnya setiap rerun): tetapkan lagi
    tab aktif bersamaan dengan klik tombol di dalamnya."""
    tombol = _tombol(at, label)
    at.session_state["tab_admin"] = tab
    tombol.click().run()
    assert not at.exception, at.exception


def test_banner_tanpa_krs(app):
    assert any("Data KRS belum tersedia" in i.value for i in app.info)
    assert any("ARM: belum aktif" in c.value for c in app.sidebar.caption)
    assert any("KRS akademik: belum diimpor" in c.value for c in app.sidebar.caption)


def test_asal_data_tidak_tampil_di_semua_peran(app_krs):
    """Asal data (asli/sintetis/fixture) tidak ditampilkan di UI, untuk setiap peran."""
    for peran in ("Mahasiswa", "DPA", "Administrator"):
        _pilih_peran(app_krs, peran)
        assert any("Riwayat multisemester: tersedia" in c.value for c in app_krs.sidebar.caption)
        elemen = [*app_krs.caption, *app_krs.markdown, *app_krs.info, *app_krs.success, *app_krs.warning]
        teks = " ".join(str(e.value) for e in elemen).lower()
        assert "sintetis" not in teks and "fixture" not in teks


def test_peran_berproteksi_butuh_kata_sandi(app):
    """KF-22: Administrator/DPA terkunci sampai kata sandi benar; salah berulang → dikunci per sesi."""
    _pilih_peran(app, "Administrator", sandi=None)
    assert not app.tabs and any(b.label == "Masuk" for b in app.button)
    for _ in range(5):
        app.text_input[0].set_value("salah")
        _tombol(app, "Masuk").click().run()
    assert any("Terlalu banyak percobaan" in e.value for e in app.error)
    log = db.baca(load_config(), "SELECT * FROM log_akses")
    assert len(log) == 5 and set(log["berhasil"]) == {0} and set(log["peran"]) == {"Administrator"}


def test_masuk_lalu_keluar(app):
    _pilih_peran(app, "DPA")
    assert not any(b.label == "Masuk" for b in app.button)
    _tombol(app, "Keluar").click().run()
    assert any(b.label == "Masuk" for b in app.button)


def test_mahasiswa_bangkitkan_kedua_mode(app, tmp_path):
    app.selectbox(key="f_agama").set_value("Islam")
    app.number_input(key="f_sem").set_value(1)
    app.number_input(key="f_ip").set_value(3.2)
    next(r for r in app.radio if r.label == "Mode").set_value("keduanya")
    _tombol(app, "Bangkitkan Rencana").click().run()
    assert not app.exception, app.exception
    assert any("Rencana tersimpan" in s.value for s in app.success)
    assert db.jumlah(load_config(), "rencana") == 2
    teks = " ".join(m.value for m in app.markdown)
    assert "Status validasi aturan domain" in teks and "Beban kognitif per semester" in teks
    # LLM hanya untuk ekstraksi beban kognitif — tidak ada penjelasan generatif
    assert not [b for b in app.button if "Jelaskan" in b.label]


def test_mahasiswa_tanpa_agama_diberi_peringatan(app):
    app.number_input(key="f_sem").set_value(1)
    next(r for r in app.radio if r.label == "Mode").set_value("arm")
    _tombol(app, "Bangkitkan Rencana").click().run()
    assert not app.exception, app.exception
    assert any("Agama mahasiswa belum diisi" in w.value for w in app.warning)


def test_mahasiswa_jalur_b_menampilkan_isian_konversi(app):
    app.selectbox(key="f_jalur").set_value("B").run()
    assert not app.exception, app.exception
    assert any("Tema pilihan yang dipenuhi lewat konversi" in m.label for m in app.multiselect)


def test_dpa_simpan_catatan(app, tmp_path):
    from src.config import load_config
    from src.planner import KonteksPlanner, muat_rencana, simpan_rencana, susun_rencana
    from src.schemas import Profile

    cfg = load_config()
    plan = susun_rencana(Profile(semester_saat_ini=1, ip_terakhir=3.0, agama="Islam"), "hybrid",
                         KonteksPlanner.muat(cfg), label="uji-dpa")
    simpan_rencana(plan, cfg)
    _pilih_peran(app, "DPA")
    app.selectbox[0].set_value(plan.id).run()
    assert any(m.label == "IPK" for m in app.metric)
    next(r for r in app.radio if r.label == "Validasi akhir DPA").set_value("perlu_revisi")
    app.text_area[0].set_value("Verifikasi syarat PLP bersama DPA.")
    _tombol(app, "Simpan validasi & catatan").click().run()
    assert not app.exception, app.exception
    tersimpan = muat_rencana(cfg, plan.id)
    assert tersimpan.catatan_dpa == "Verifikasi syarat PLP bersama DPA." and tersimpan.status_dpa == "perlu_revisi"
    # UC-04 alur 6: mahasiswa melihat status & catatan lewat kode rencana
    _pilih_peran(app, "Mahasiswa")
    app.text_input(key="cek_pid").set_value(plan.id).run()
    assert not app.exception, app.exception
    assert any("perlu revisi" in w.value for w in app.warning)
    assert any("Verifikasi syarat PLP" in m.value for m in app.markdown)
    app.text_input(key="cek_pid").set_value("tidak-ada").run()
    assert any("tidak ditemukan" in e.value for e in app.error)


def test_mahasiswa_riwayat_nilai_menghitung_ipk(app):
    """KF-12/KD-10: riwayat nilai → MK lulus/belum lulus & IPK. COM60014 A (5 SKS), CIE61301 B (3 SKS),
    COM60015 E (2 SKS) → lulus 2 MK (8 SKS); IPK (20 + 9 + 0)/10 = 2,90."""
    import pandas as pd

    app.session_state["f_riwayat"] = pd.DataFrame([
        {"Mata kuliah": "COM60014 - Pemrograman Dasar", "Nilai": "A"},
        {"Mata kuliah": "CIE61301 - Kalkulus dan Aljabar Linear", "Nilai": "B"},
        {"Mata kuliah": "COM60015 - Matematika Diskrit", "Nilai": "E"}])
    app.run()
    assert not app.exception, app.exception
    nilai = {m.label: m.value for m in app.metric}
    assert nilai["MK lulus"] == "2" and nilai["SKS lulus"] == "8" and nilai["IPK"] == "2.90"
    assert any("COM60015" in c.value for c in app.caption)
    app.selectbox(key="f_agama").set_value("Islam")
    app.number_input(key="f_sem").set_value(3)
    next(r for r in app.radio if r.label == "Mode").set_value("arm")
    _tombol(app, "Bangkitkan Rencana").click().run()
    assert not app.exception, app.exception
    plan = app.session_state["rencana_terakhir"]["arm"]
    assert plan.profil.riwayat_nilai == {"COM60014": "A", "CIE61301": "B", "COM60015": "E"}
    assert plan.profil.mk_pernah_gagal == {"COM60015"}


def test_admin_tab_dan_penilaian_bloom_mock(app, tmp_path):
    _pilih_peran(app, "Administrator")
    label_tab = [t.label for t in app.tabs]
    for t in ["Mulai di sini", "Kurikulum", "Data akademik", "Preprocessing", "ARM", "Beban kognitif (LLM)", "Parameter",
              "Simulasi & validasi", "Log"]:
        assert t in label_tab
    assert not any(s.label == "Provider" for s in app.selectbox)          # tab lain belum dijalankan (lazy)
    _buka_tab(app, "Kurikulum")
    teks = " ".join(m.value for m in app.markdown)
    assert "CPMK yang perlu ditinjau" in teks and "Override prasyarat" in teks
    _buka_tab(app, "Beban kognitif (LLM)")
    next(s for s in app.selectbox if s.label == "Provider").set_value("mock")
    _klik_di_tab(app, "Beban kognitif (LLM)", "Jalankan penilaian Bloom")
    assert not app.exception, app.exception
    assert any("permintaan API" in s.value for s in app.success)
    assert db.jumlah(load_config(), "batch_llm") == 3                       # 3 pengulangan × 1 batch
    assert db.jumlah(load_config(), "rating_bloom", "sumber = 'llm_mock'") > 0
    teks = " ".join(m.value for m in app.markdown)
    assert "RM1" in teks
    assert any("BUKAN hasil penelitian" in w.value for w in app.warning)        # LLM mock pada RM1


def test_admin_preprocessing_arm_simulasi_dengan_krs_fixture(app_krs, tmp_path):
    _pilih_peran(app_krs, "Administrator")
    _buka_tab(app_krs, "Preprocessing")
    _klik_di_tab(app_krs, "Preprocessing", "Jalankan preprocessing")
    assert not app_krs.exception, app_krs.exception
    assert db.jumlah(load_config(), "transaksi", "split = 'latih'") > 0
    _buka_tab(app_krs, "ARM")
    _klik_di_tab(app_krs, "ARM", "Jalankan FP-Growth")
    assert not app_krs.exception, app_krs.exception
    assert any("aturan dari" in s.value for s in app_krs.success)
    _buka_tab(app_krs, "Simulasi & validasi")
    _klik_di_tab(app_krs, "Simulasi & validasi", "Jalankan simulasi komparasi")
    assert not app_krs.exception, app_krs.exception
    assert (tmp_path / "ekspor" / "simulasi_metrik.csv").exists() and db.baca_ringkasan(load_config(), "simulasi")


def _demo_segar(tmp_path):
    """Riwayat pelengkap (src/sintetis) + preprocessing → transaksi uji segar."""
    from src.preprocessing import jalankan_preprocessing
    from src.sintetis import generate
    from tests.conftest import buat_cfg

    cfg = buat_cfg(tmp_path, generator={"n_alumni": 30})
    shutil.rmtree(cfg.raw_dir)
    generate(cfg, cfg.raw_dir)
    jalankan_preprocessing(cfg)


def test_mahasiswa_terapkan_profil_alumni_uji(tmp_path, monkeypatch):
    """Regresi: 'Terapkan' mengisi MK lulus semester 1..k (kode Kurikulum 2024) sehingga rencana dimulai
    di semester k+1 tanpa MK paket semester awal sebagai MK tertunda."""
    at = _app(tmp_path, monkeypatch, dengan_krs=False, siapkan=_demo_segar)
    at.slider(key="demo_k").set_value(4)
    _tombol(at, "Terapkan").click().run()
    assert not at.exception, at.exception
    riwayat = at.session_state["f_riwayat"]
    lulus = {x.split(" - ")[0] for x, n in zip(riwayat["Mata kuliah"], riwayat["Nilai"]) if n != "E"}
    assert at.session_state["f_sem"] == 5 and len(lulus) >= 25
    assert any("diterapkan" in s.value and "rencana mulai semester 5" in s.value for s in at.success)
    assert next(m for m in at.metric if m.label == "IPK").value != "–"        # nilai alumni ikut terbawa
    at.selectbox(key="f_agama").set_value("Islam").run()      # baris MK agama riwayat ikut disesuaikan
    _tombol(at, "Bangkitkan Rencana").click().run()
    assert not at.exception, at.exception
    judul = [m.value for m in at.markdown if m.value.startswith("*Semester")]
    assert judul and judul[0].startswith("*Semester 5")
    tabel = pd.concat(d.value for d in at.dataframe if "Kode" in d.value.columns and "Prioritas" in d.value.columns)
    assert not set(tabel["Kode"]) & set(lulus)


def test_baris_agama_mengikuti_kolom_agama(app):
    """Kolom Agama mengganti baris MK agama di riwayat nilai (nilai terbaik dipertahankan, tidak ganda)."""
    kur = load_kurikulum_uji()
    app.session_state["f_riwayat"] = pd.DataFrame(
        [{"Mata kuliah": kur.label("MPK60002"), "Nilai": "A"}, {"Mata kuliah": kur.label("COM60016"), "Nilai": "B"},
         {"Mata kuliah": kur.label("MPK60004"), "Nilai": "C"}], columns=["Mata kuliah", "Nilai"])
    app.selectbox(key="f_agama").set_value("Islam").run()
    assert not app.exception, app.exception
    df = app.session_state["f_riwayat"]
    agama = [(mk.split(" - ")[0], n) for mk, n in zip(df["Mata kuliah"], df["Nilai"]) if mk.startswith("MPK6000")]
    assert agama == [("MPK60001", "A")] and len(df) == 2
    assert any("Baris MK agama disesuaikan" in s.value for s in app.success)
    app.selectbox(key="f_agama").set_value("Islam").run()           # sudah sesuai → tidak berubah lagi
    assert len(app.session_state["f_riwayat"]) == 2


def load_kurikulum_uji():
    from src.data_loader import load_kurikulum
    return load_kurikulum(load_config())


def test_mahasiswa_simulasi_skenario_molor(app):
    """Tombol simulasi skenario mengisi formulir dan menampilkan perbandingan perkiraan lulus."""
    at = app
    at.selectbox(key="skenario_id").set_value("tertinggal")
    _tombol(at, "Jalankan simulasi skenario").click().run()
    assert not at.exception, at.exception
    assert at.session_state["f_sem"] == 5 and at.session_state["f_agama"] == "Islam"
    label = [m.label for m in at.metric]
    assert "Pola konvensional: perkiraan lulus" in label
    assert any(lb.startswith("Sistem") and lb.endswith("perkiraan lulus") for lb in label)
    _tombol(at, "Bangkitkan Rencana").click().run()
    assert not at.exception, at.exception
    _tombol(at, "Tutup hasil simulasi").click().run()
    assert "Pola konvensional: perkiraan lulus" not in [m.label for m in at.metric]


def test_mahasiswa_transaksi_uji_basi_diberi_peringatan(tmp_path, monkeypatch):
    """Transaksi uji basi (mis. KRS sintetis lama berkode non-2024) tidak dipakai diam-diam."""
    def basi(base):
        from src.data_loader import simpan_transaksi
        from src.db import simpan_ringkasan
        from src.schemas import Transaksi
        from tests.conftest import buat_cfg

        cfg = buat_cfg(base)
        simpan_transaksi(cfg, "uji", [Transaksi(id_anon="A1", angkatan=2020, semester_ke=s, ip_semester=3.0,
                                                items=[f"PTI10{s}@S{s}"]) for s in (1, 2, 3)])
        simpan_ringkasan(cfg, "preprocessing", {"sidik_data": "sidik-lama"})

    at = _app(tmp_path, monkeypatch, dengan_krs=False, siapkan=basi)
    assert any("Profil alumni uji tidak dapat dipakai" in w.value and "basi" in w.value for w in at.warning)
    assert not [s for s in at.slider if s.key == "demo_k"]


def test_mahasiswa_pilihan_tema_dan_geser_mk(app, tmp_path):
    """Iterasi 3: panel MK pilihan per tema → preferensi masuk profil; menu pindah (alternatif drag & drop)
    menggeser MK, menyimpan rencana yang disesuaikan, dan menampilkan ringkasan penyesuaian."""
    from src.penyesuaian import semester_tujuan_sah
    from src.planner import muat_rencana

    _tombol(app, "Atur MK pilihan per tema").click().run()
    assert not app.exception, app.exception
    assert sum(s.label.startswith("Tema ") for s in app.selectbox) == 9
    assert any(b.label == "Simpan pilihan tema" for b in app.button)
    # klik di dalam st.dialog (fragmen) tidak didukung AppTest → pilihan diisikan lewat session state
    app.session_state["f_pilihan_tema"] = {"A": "CIE61403"}
    app.run()
    assert any("Tema A → CIE61403" in c.value for c in app.caption)

    app.selectbox(key="f_agama").set_value("Islam")
    app.number_input(key="f_sem").set_value(3)
    next(r for r in app.radio if r.label == "Mode").set_value("arm")
    _tombol(app, "Bangkitkan Rencana").click().run()
    assert not app.exception, app.exception
    plan = app.session_state["rencana_terakhir"]["arm"]
    assert plan.profil.pilihan_tema == {"A": "CIE61403"}

    from src.planner import KonteksPlanner
    kp = KonteksPlanner.muat(load_config())
    kode, ke = next((k, sah[0]) for sm in plan.semester if sm.jenis == "reguler" for k in [i.kode for i in sm.items]
                    if (sah := semester_tujuan_sah(plan, k, kp)))
    app.selectbox(key="menu-mk-mhs-0").set_value(kode).run()
    app.selectbox(key="menu-ke-mhs-0").set_value(ke)
    _tombol(app, "Pindahkan").click().run()
    assert not app.exception, app.exception
    baru = app.session_state["rencana_terakhir"]["arm"]
    assert baru.disesuaikan and baru.penyesuaian[0].kode == kode and baru.penyesuaian[0].ke == ke
    assert muat_rencana(load_config(), plan.id).disesuaikan
    assert any("Pemindahan diterima" in s.value for s in app.success)
    assert any(b.label == "Kembalikan ke rekomendasi sistem" for b in app.button)
    _tombol(app, "Kembalikan ke rekomendasi sistem").click().run()
    assert not app.exception, app.exception
    assert not app.session_state["rencana_terakhir"]["arm"].disesuaikan


def test_admin_mulai_di_sini_dan_atur_ulang(app_krs, tmp_path):
    """Iterasi 3: tab Mulai di sini (daftar periksa, templat) dan atur ulang dengan konfirmasi ketik."""
    from src.preprocessing import jalankan_preprocessing

    at = app_krs
    jalankan_preprocessing(load_config())
    _pilih_peran(at, "Administrator")
    assert at.tabs[0].label == "Mulai di sini"
    assert any("Langkah wajib selesai" in (p.text or "") for p in at.get("progress"))
    assert any(b.label == "Atur ulang sistem ke kondisi awal" and b.disabled for b in at.button)
    at.text_input(key="konfirmasi_atur_ulang").set_value("ATUR ULANG").run()
    _tombol(at, "Atur ulang sistem ke kondisi awal").click().run()
    assert not at.exception, at.exception
    assert any("Sistem diatur ulang" in s.value for s in at.success)
    assert db.jumlah(load_config(), "transaksi") == 0
    assert not (tmp_path / "raw" / "histori_krs.csv").exists()

