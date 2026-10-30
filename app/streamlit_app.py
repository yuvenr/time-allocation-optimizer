"""UI Streamlit Sistem Rekomendasi Rencana Studi PTI UB — Kurikulum 2024 (ARM + LLM), versi 2.

Jalankan dari root proyek:  streamlit run app/streamlit_app.py

Peran (KF-22: Administrator & DPA berkata sandi — hash PBKDF2 di .env, lihat ``src/auth.py``):
- Mahasiswa     (UC-01..03): profil (jalur, agama, riwayat nilai/IPK) → rencana → beban, validasi, alasan;
                             cek status validasi DPA lewat kode rencana.
- DPA           (UC-04, 05): daftar rencana tersimpan → detail (MK diambil, nilai) → validasi akhir + catatan.
- Administrator (UC-06..12): kurikulum, data akademik, preprocessing, ARM, beban kognitif (LLM),
                             parameter, simulasi & validasi, log.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.berkas import pulihkan as pulihkan_berkas, simpan as simpan_berkas  # noqa: E402
from src.config import CONFIG_PATH, OVERRIDE_PATH, load_config, reset_override, save_override  # noqa: E402
from src import auth, db  # noqa: E402
from src.data_loader import (FILE_KURIKULUM, KOLOM_KRS_WAJIB, ada_transaksi, baca_csv_kurikulum,  # noqa: E402
                             histori_segar, load_aturan, load_kurikulum, status_data_krs, sumber_data,
                             transaksi_segar)
from src.logutil import baca_log  # noqa: E402
from src.pipeline import hitung_dan_simpan_theta, status_pipeline  # noqa: E402
from src.planner import (KonteksPlanner, daftar_rencana, muat_rencana,  # noqa: E402
                         perbarui_catatan_dpa, simpan_rencana, susun_rencana)
from src.preprocessing import profil_dari_riwayat  # noqa: E402
from src.rules import hitung_ipk, tema_terpenuhi  # noqa: E402
from src.schemas import JALUR, Profile  # noqa: E402
from src.simulate import bentuk_profil, kelompokkan, riwayat_alumni  # noqa: E402
from src.templat import baca_csv_unggahan  # noqa: E402
from tampilan import LABEL_MODE, LABEL_SINGKAT, bandingkan, tampilkan_rencana  # noqa: E402

st.set_page_config(page_title="Time Allocation Optimizer", layout="wide")

BELUM_DIISI = "(belum diisi)"
LABEL_STATUS_DPA = {"belum_ditinjau": "belum ditinjau", "disetujui": "disetujui", "perlu_revisi": "perlu revisi"}
# kunci log yang memuat asal data — tetap disimpan di basis data & log, tidak ditampilkan
KUNCI_ASAL_DATA = {"data_krs", "sumber_data", "sumber_riwayat", "sumber_per_semester", "n_aturan_per_sumber",
                   "transaksi_per_semester_sumber", "sumber", "sumber_kalibrasi"}


# ---------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------


def _versi(cfg) -> str:
    """Versi data (kunci cache): berubah bila data masukan, hasil proses, rating, atau konfigurasi berubah. Dihitung
    sekali per tampilan (satu kueri ke basis data); status yang jarang berubah di-cache per versi agar tiap klik
    tidak mengulang belasan kueri ke basis data cloud."""
    files = [cfg.raw_dir / "histori_krs.csv", CONFIG_PATH, OVERRIDE_PATH]
    return "|".join([db.keterangan(cfg)] + [f"{f}:{f.stat().st_mtime_ns}" if f.exists() else f"{f}:-" for f in files]
                    + [db.sidik_konteks(cfg)])


def versi_data() -> str:
    return st.session_state["_versi_data"]


@st.cache_resource(show_spinner=False)
def _konteks(versi: str) -> KonteksPlanner:
    return KonteksPlanner.muat(load_config())


@st.cache_data(show_spinner=False)
def _status_data(versi: str) -> dict:
    cfg = load_config()
    sd = sumber_data(cfg)
    return {"sumber": sd, "akademik": (db.baca_ringkasan(cfg, "akademik") or {}) if sd["krs_akademik"] else {}}


@st.cache_data(show_spinner=False)
def _status_pipeline(versi: str) -> dict:
    return status_pipeline(load_config())


@st.cache_resource(show_spinner=False)
def _alumni_uji(versi: str) -> tuple[bool, str | None, dict]:
    """(ada transaksi, alasan bila basi, kelompok transaksi uji per alumni)."""
    cfg = load_config()
    if not ada_transaksi(cfg):
        return False, None, {}
    uji, alasan = transaksi_segar(cfg, "uji")
    return True, alasan, ({} if alasan else kelompokkan(uji))


@st.cache_data(show_spinner=False)
def _formulir_pakar_csv(versi: str) -> bytes:
    from src.export import formulir_pakar

    return formulir_pakar(load_config()).to_csv(index=False).encode("utf-8-sig")


@st.cache_data(show_spinner=False)
def _zip_paket_data(versi: str) -> bytes:
    """Zip paket data uji (dibangun ulang hanya bila versi data berubah; ValueError bila data belum lengkap)."""
    from src.paket_uji import buat_paket, raw_paket, zip_paket

    cfg = load_config()
    return zip_paket(buat_paket(cfg), raw_paket(cfg))


@st.cache_data(show_spinner=False, ttl=20)
def _kesiapan(versi: str) -> list:
    """Daftar periksa (tab Mulai di sini): per versi data, disegarkan paling lambat 20 detik (langkah 12 menghitung
    rencana & validasi DPA yang tidak mengubah versi data)."""
    from src.kesiapan import periksa_kesiapan

    return periksa_kesiapan(load_config())


@st.cache_data(show_spinner=False)
def _cadangan_rating(versi: str) -> pd.DataFrame:
    from src.llm.beban import ekspor_rating_llm

    cfg = load_config()
    return ekspor_rating_llm(cfg) if db.ada_db(cfg) else pd.DataFrame()


def banner(cfg) -> None:
    if not any(_status_data(versi_data())["sumber"].values()):
        st.info("**Data KRS belum tersedia, ARM belum aktif.** Rekomendasi disusun dari struktur kurikulum "
                "resmi (08, 09) dan aturan domain. Administrator dapat mengimpor data akademik.")


def gerbang_peran(peran: str, cfg) -> bool:
    """KF-22: True bila peran boleh dibuka di sesi ini (Mahasiswa selalu; Administrator/DPA setelah masuk)."""
    if not auth.perlu_masuk(peran, cfg) or st.session_state.get(f"masuk_{peran}"):
        return True
    st.header(f"Masuk sebagai {peran}")
    if auth.hash_peran(peran) is None:
        st.error(f"Kata sandi peran {peran} belum dikonfigurasi. Jalankan `python -m src.auth`, salin hash ke "
                 f"`.env` ({auth.VAR_HASH[peran]}), lalu muat ulang aplikasi.")
        return False
    gagal = st.session_state.get(f"gagal_{peran}", 0)
    if gagal >= cfg.auth.maks_gagal:
        st.error("Terlalu banyak percobaan gagal. Muat ulang halaman untuk mencoba lagi.")
        return False
    with st.form(f"form_masuk_{peran}"):
        pw = st.text_input("Kata sandi", type="password")
        if st.form_submit_button("Masuk", type="primary"):
            if auth.masuk(peran, pw, cfg):
                st.session_state[f"masuk_{peran}"] = True
                st.rerun()
            st.session_state[f"gagal_{peran}"] = gagal + 1
            st.error("Terlalu banyak percobaan gagal. Muat ulang halaman untuk mencoba lagi."
                     if gagal + 1 >= cfg.auth.maks_gagal else "Kata sandi salah.")
    return False


# ---------------------------------------------------------------------
# Mahasiswa (UC-01..UC-03)
# ---------------------------------------------------------------------


KOLOM_RIWAYAT = ["Mata kuliah", "Nilai"]


def _set_riwayat(riwayat: dict[str, str], kur) -> None:
    """Isi tabel riwayat nilai (dipakai CSV, profil alumni demo) lalu paksa editor dibuat ulang."""
    st.session_state["f_riwayat"] = pd.DataFrame(
        [{"Mata kuliah": kur.label(k), "Nilai": n} for k, n in sorted(riwayat.items())], columns=KOLOM_RIWAYAT)
    st.session_state["riwayat_versi"] = st.session_state.get("riwayat_versi", 0) + 1


def _baca_riwayat(df: pd.DataFrame, kur, cfg) -> tuple[dict[str, str], list[str]]:
    """Tabel editor → riwayat {kode: nilai terbaik} + daftar catatan (baris tidak lengkap, MK ganda)."""
    peta = {kur.label(k): k for k in kur.kode}
    skala = cfg.aturan.SKALA_NILAI
    riwayat: dict[str, str] = {}
    catatan: list[str] = []
    for r in df.itertuples(index=False):
        mk, nilai = r[0], r[1]
        if not isinstance(mk, str) or not isinstance(nilai, str) or not mk or not nilai:
            if isinstance(mk, str) and mk:
                catatan.append(f"{mk}: nilai belum diisi, diabaikan")
            continue
        kode = peta.get(mk)
        if kode is None:
            continue
        if kode in riwayat:
            catatan.append(f"{kode} diisi lebih dari sekali, dipakai nilai terbaik")
            if skala.index(nilai) >= skala.index(riwayat[kode]):
                continue
        riwayat[kode] = nilai
    return riwayat, catatan


def _selaraskan_agama(df: pd.DataFrame, agama: str, kur, cfg) -> tuple[pd.DataFrame, str] | None:
    """Baris MK agama di tabel riwayat mengikuti kolom Agama: semua baris grup AGAMA diganti satu baris
    MK agama terpilih dengan nilai terbaiknya. None bila sudah sesuai (atau agama belum dipilih)."""
    target = kur.mk_agama(None if agama == BELUM_DIISI else agama)
    if target is None:
        return None
    peta = {kur.label(k): k for k in kur.grup("AGAMA")}
    baris = [i for i, mk in enumerate(df["Mata kuliah"]) if mk in peta]
    if not baris or (len(baris) == 1 and peta[df["Mata kuliah"].iloc[baris[0]]] == target):
        return None
    skala = cfg.aturan.SKALA_NILAI
    nilai = [n for n in df["Nilai"].iloc[baris] if isinstance(n, str) and n in skala]
    lama = sorted({peta[df["Mata kuliah"].iloc[i]] for i in baris})
    baru = df.copy()
    baru.iloc[baris[0], baru.columns.get_loc("Mata kuliah")] = kur.label(target)
    baru.iloc[baris[0], baru.columns.get_loc("Nilai")] = min(nilai, key=skala.index) if nilai else None
    baru = baru.drop(index=baru.index[baris[1:]]).reset_index(drop=True)
    return baru, f"Baris MK agama disesuaikan dengan agama {agama}: {', '.join(lama)} → {target} (nilai tetap)."


def _isi_dari_riwayat(cfg, ctx) -> None:
    with st.expander("Isi otomatis dari berkas riwayat nilai (CSV `kode_mk,nilai_huruf`; kode 2020 dinormalisasi)"):
        _tombol_templat("riwayat_nilai", "mhs")
        up = st.file_uploader("Riwayat nilai", type=["csv"], key="up_riwayat",
                              help="Tidak perlu NIM/nama. Kode Kurikulum 2020 dipetakan lewat 11_penyetaraan (AT21).")
        if up is not None and st.button("Terapkan riwayat"):
            try:
                df = baca_csv_unggahan(up, dtype=str)
                if not {"kode_mk", "nilai_huruf"} <= set(df.columns):
                    raise ValueError("kolom wajib: kode_mk, nilai_huruf")
                riwayat = dict(zip(df["kode_mk"].str.strip(), df["nilai_huruf"].str.strip()))
                prof, lap = profil_dari_riwayat(riwayat, ctx.kurikulum, cfg, semester_saat_ini=1, ip_terakhir=0.0)
            except Exception as e:
                st.error(f"Riwayat ditolak: {e}")
            else:
                _set_riwayat(prof.riwayat_nilai, ctx.kurikulum)
                pesan = [f"{len(prof.mk_lulus)} MK lulus, {len(prof.mk_pernah_gagal)} MK belum lulus "
                         f"(ambang lulus {cfg.aturan.NILAI_LULUS_MINIMUM})."]
                for kunci, judul in (("dipetakan", "Dipetakan"), ("dihapus", "MK 2020 dihapus (tidak dihitung)"),
                                     ("tidak_dikenal", "Kode tidak dikenal")):
                    if lap[kunci]:
                        pesan.append(f"{judul}: {', '.join(lap[kunci])}")
                st.session_state["demo_info"] = " ".join(pesan)
                st.rerun()


def _isi_dari_alumni_uji(cfg, ctx) -> None:
    ada, alasan, kelompok = _alumni_uji(versi_data())
    if not ada:
        return
    with st.expander("Isi otomatis dari profil alumni data uji (simulasi)"):
        if alasan:
            # transaksi basi (mis. KRS lama berkode non-Kurikulum 2024) tidak dipakai: kodenya
            # tidak dikenal kurikulum sehingga MK lulus akan hilang dari isian tanpa pemberitahuan
            st.warning(f"Profil alumni uji tidak dapat dipakai: {alasan}. Jalankan ulang Preprocessing "
                       f"(Administrator → Preprocessing) atau `python -m src.pipeline`.")
            return
        if not kelompok:
            st.caption("Data uji tidak memuat alumni.")
            return
        c1, c2, c3 = st.columns([2, 1, 1])
        alumni = c1.selectbox("Alumni (ID anonim)", list(kelompok), key="demo_alumni")
        k = c2.slider("Potong riwayat di semester k", 1, 7, 2, key="demo_k")
        if c3.button("Terapkan", width="stretch"):
            kode = set(ctx.kurikulum.kode)
            hasil = bentuk_profil(kelompok[alumni], k, cfg.kurikulum.horizon_semester, kode)
            if hasil is None:
                st.error("Alumni ini tidak memiliki riwayat setelah semester tersebut.")
            else:
                prof, _ = hasil
                histori, _ = histori_segar(cfg, "uji")
                riwayat = riwayat_alumni(histori[histori["id_anon"] == alumni], k, kode, cfg)
                _set_riwayat(riwayat, ctx.kurikulum)
                st.session_state.update(f_sem=prof.semester_saat_ini, f_ip=float(prof.ip_terakhir))
                st.session_state["demo_info"] = (
                    f"Riwayat {alumni} semester 1–{k} diterapkan: {len(prof.mk_lulus)} MK lulus "
                    f"({ctx.kurikulum.total_sks(prof.mk_lulus)} SKS), {len(prof.mk_pernah_gagal)} MK belum lulus, "
                    f"IP semester {k} = {prof.ip_terakhir:.2f}; rencana mulai semester {prof.semester_saat_ini}.")
                st.rerun()


def _simulasi_skenario(cfg, ctx) -> None:
    """Demo K5: profil fiktif berisiko molor > 8 semester → pola konvensional vs rekomendasi sistem."""
    from src.skenario import SKENARIO, cari, jalankan_skenario

    with st.expander("Simulasi mahasiswa berisiko studi molor (> 8 semester)"):
        st.caption("Profil fiktif Jalur A dengan permasalahan studi (MK tidak lulus, tertinggal paket, IP rendah). "
                   "Sistem dibandingkan dengan pola KRS konvensional (mengambil paket semester berjalan dulu, "
                   "lalu MK tertunda bila SKS masih cukup) dengan aturan akademik yang sama.")
        pilih = st.selectbox("Skenario", [s.id for s in SKENARIO], key="skenario_id",
                             format_func=lambda i: cari(i).nama)
        sk = cari(pilih)
        st.markdown(f"{sk.deskripsi} Semester berjalan **S{sk.semester}**, IP semester lalu **{sk.ip:.2f}**.")
        if st.button("Jalankan simulasi skenario", type="primary"):
            with st.spinner("Menyusun rencana…"):
                hasil = jalankan_skenario(sk, ctx)
            prof = hasil["profil"]
            _set_riwayat(prof.riwayat_nilai, ctx.kurikulum)
            st.session_state.update(f_sem=prof.semester_saat_ini, f_ip=float(prof.ip_terakhir),
                                    f_jalur=prof.jalur, f_agama=prof.agama, skenario_hasil={"skenario": sk, **hasil})
            st.session_state.pop("rencana_terakhir", None)
            st.rerun()


def _hasil_skenario(ctx) -> None:
    hasil = st.session_state.get("skenario_hasil")
    if not hasil:
        return
    kur, sk, lulus = ctx.kurikulum, hasil["skenario"], hasil["lulus"]

    def teks(s):
        return f"S{s}" if s is not None else f"> S{kur.param_int('AT04', 'semester_max')}"

    with st.container(border=True):
        st.markdown(f"**Hasil simulasi: {sk.nama}.** Profil sudah diisikan ke formulir di bawah; klik "
                    f"*Bangkitkan Rencana* untuk melihat rencana lengkap beserta alasannya.")
        kolom = ["konvensional"] + list(hasil["rencana"])
        c = st.columns(len(kolom))
        judul = {"konvensional": "Pola konvensional", **{m: f"Sistem ({LABEL_SINGKAT[m]})" for m in hasil["rencana"]}}
        for i, k in enumerate(kolom):
            selisih = None
            if k != "konvensional" and None not in (lulus[k], lulus["konvensional"]):
                selisih = lulus[k] - lulus["konvensional"]
            c[i].metric(f"{judul[k]}: perkiraan lulus", teks(lulus[k]),
                        delta=f"{selisih:+d} semester" if selisih else None, delta_color="inverse")
        acuan = hasil["rencana"].get("hybrid") or hasil["rencana"]["arm"]
        sistem = {s.semester: [i.kode for i in s.items] for s in acuan.semester}
        konv = hasil["konvensional"].semester
        baris = []
        for s in sorted(set(sistem) | set(konv)):
            a, b = konv.get(s, []), sistem.get(s, [])
            baris.append({"Semester": f"S{s}", "Konvensional (SKS)": kur.total_sks(a),
                          "Konvensional": ", ".join(a) or "–", f"Sistem {LABEL_SINGKAT[acuan.mode]} (SKS)":
                          kur.total_sks(b), f"Sistem {LABEL_SINGKAT[acuan.mode]}": ", ".join(b) or "–"})
        st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch")
        st.caption("Asumsi kedua pola: IP semester mendatang tetap, semua MK yang diambil lulus, prasyarat/"
                   "penawaran/pita SKS sama (constraint engine yang sama).")
        if st.button("Tutup hasil simulasi"):
            st.session_state.pop("skenario_hasil", None)
            st.rerun()


def _cek_status_rencana(cfg) -> None:
    """UC-04 alur 6: mahasiswa melihat catatan & validasi akhir DPA atas rencananya (lewat kode rencana)."""
    with st.expander("Cek status validasi DPA atas rencana saya"):
        pid = st.text_input("Kode rencana", key="cek_pid", placeholder="mis. 20261005-101500-hybrid-ab12",
                            help="Kode ditampilkan setelah rencana dibangkitkan.")
        if not pid.strip():
            return
        try:
            plan = muat_rencana(cfg, pid.strip())
        except KeyError:
            st.error("Rencana tidak ditemukan.")
            return
        teks = LABEL_STATUS_DPA[plan.status_dpa]
        {"disetujui": st.success, "perlu_revisi": st.warning}.get(plan.status_dpa, st.info)(
            f"Status validasi DPA: **{teks}**")
        if plan.catatan_dpa:
            st.markdown(f"**Catatan DPA:** {plan.catatan_dpa}")
        if plan.catatan_dpa_diperbarui:
            st.caption(f"Ditinjau {plan.catatan_dpa_diperbarui:%d-%m-%Y %H:%M}.")


def _label_opsi_pilihan(kode: str, ctx: KonteksPlanner) -> str:
    kur = ctx.kurikulum
    m = kur.mk[kode]
    bagian = [m.nama, "ditawarkan S" + "/".join(map(str, m.semester_tabel11)) if m.semester_tabel11 else
              "semester penawaran tidak tercantum"]
    if syarat := kur.prasyarat_nilai(kode):
        bagian.append("prasyarat " + ", ".join(kur.mk[p.kode_syarat].nama for p in syarat if p.kode_syarat in kur))
    if (cl := ctx.cl_mk(kode)) is not None:
        bagian.append(f"CL {cl:.1f}")
    return f"{kode} · " + " · ".join(bagian)


@st.dialog("MK pilihan per tema", width="large")
def _panel_pilihan_tema(ctx: KonteksPlanner, lulus: set[str]) -> None:
    """Panel (drawer) iterasi 3: mahasiswa memilih MK pilihan yang diminati untuk setiap tema; tema yang
    dibiarkan 'Rekomendasi sistem' dipilihkan planner (ARM, penawaran, beban)."""
    from src.planner import preferensi_sah

    kur = ctx.kurikulum
    st.caption("Setiap tema wajib ditempuh minimal 3 SKS (satu MK). Pilih MK yang sesuai minat Anda, atau biarkan "
               "**Rekomendasi sistem**. MK yang dipilih tetap harus memenuhi prasyarat dan semester penawarannya; "
               "bila tidak dapat dijadwalkan, rencana memberi peringatan.")
    awal = st.session_state.get("f_pilihan_tema", {})
    selesai = tema_terpenuhi(lulus, (), kur)
    baru: dict[str, str] = {}
    for t, tp in sorted(kur.tema.items()):
        nama = ctx.cfg.tampilan.nama_tema.get(t, tp.nama)
        opsi = sorted(k for k in kur.mk_tema(t) if preferensi_sah({t: k}, kur)[0])
        if t in selesai:
            st.markdown(f"**Tema {t}: {nama}.** Sudah terpenuhi "
                        f"({', '.join(kur.label(k) for k in sorted(kur.mk_tema(t) & lulus))})")
            continue
        pilih = st.selectbox(f"Tema {t}: {nama}", [""] + opsi,
                             index=([""] + opsi).index(awal[t]) if awal.get(t) in opsi else 0,
                             format_func=lambda k: "Rekomendasi sistem" if not k else _label_opsi_pilihan(k, ctx))
        if pilih:
            baru[t] = pilih
    c1, c2 = st.columns(2)
    if c1.button("Simpan pilihan tema", type="primary", width="stretch"):
        st.session_state["f_pilihan_tema"] = baru
        st.rerun()
    if c2.button("Serahkan semua ke sistem", width="stretch"):
        st.session_state["f_pilihan_tema"] = {}
        st.rerun()


def halaman_mahasiswa(cfg, ctx: KonteksPlanner) -> None:
    kur = ctx.kurikulum
    st.header("Rencana studi mahasiswa")
    st.caption(f"Isi profil akademik dan riwayat nilai, pilih mode, lalu bangkitkan rencana dari semester berjalan "
               f"hingga semester {cfg.kurikulum.horizon_semester} (diperpanjang bila belum lulus, maksimal "
               f"{kur.param_int('AT04', 'semester_max')} semester).")
    ada_theta = ctx.theta is not None
    _cek_status_rencana(cfg)
    _isi_dari_riwayat(cfg, ctx)
    _isi_dari_alumni_uji(cfg, ctx)
    _simulasi_skenario(cfg, ctx)
    if info := st.session_state.pop("demo_info", None):
        st.success(info)
    _hasil_skenario(ctx)

    for kunci, awal in (("f_sem", 1), ("f_ip", 3.0), ("f_jalur", "A"), ("f_agama", BELUM_DIISI)):
        st.session_state.setdefault(kunci, awal)
    nama_jalur = {s.jalur: s.nama_jalur for s in kur.struktur}
    c1, c2 = st.columns(2)
    jalur = c1.selectbox("Jalur kurikulum", list(JALUR), key="f_jalur",
                         format_func=lambda j: f"{j}: {nama_jalur.get(j, '')}",
                         help="Ditetapkan di awal semester 3 (AT11). Semester 1–5 identik untuk semua jalur.")
    agama = c2.selectbox("Agama (untuk MK agama)", [BELUM_DIISI] + kur.pilihan_agama, key="f_agama",
                         help="Input mahasiswa; sistem tidak menyimpulkan agama. Baris MK agama pada riwayat nilai "
                              "otomatis disesuaikan dengan pilihan ini (nilainya tetap), dan MK agama yang belum "
                              "ditempuh direncanakan sesuai pilihan ini.")

    # KF-12 / KD-10: riwayat nilai per MK → MK lulus/belum lulus, IPK (AD-10), porsi D/D+ (AD-07)
    st.markdown("**Riwayat nilai** (MK yang sudah ditempuh beserta nilai akhir terbaik)")
    opsi = [kur.label(k) for k in sorted(kur.kode, key=lambda k: (kur.mk[k].semester_acuan or 99, k))]
    awal = st.session_state.get("f_riwayat", pd.DataFrame(columns=KOLOM_RIWAYAT))
    tabel = st.data_editor(
        awal, key=f"ed_riwayat_{st.session_state.get('riwayat_versi', 0)}", num_rows="dynamic", hide_index=True,
        width="stretch", column_config={
            "Mata kuliah": st.column_config.SelectboxColumn("Mata kuliah", options=opsi, required=True, width="large"),
            "Nilai": st.column_config.SelectboxColumn("Nilai akhir", options=cfg.aturan.SKALA_NILAI, required=True)})
    if sesuai := _selaraskan_agama(tabel, agama, kur, cfg):
        st.session_state["f_riwayat"], st.session_state["demo_info"] = sesuai
        st.session_state["riwayat_versi"] = st.session_state.get("riwayat_versi", 0) + 1
        st.rerun()
    riwayat, catatan = _baca_riwayat(tabel, kur, cfg)
    lulus = {k for k, n in riwayat.items() if cfg.aturan.lulus(n)}
    gagal = set(riwayat) - lulus
    ipk, ket_ipk = hitung_ipk(riwayat, kur, cfg)
    sks_d = kur.total_sks(k for k, n in riwayat.items() if n in cfg.aturan.NILAI_DIBATASI)
    c = st.columns(4)
    c[0].metric("MK lulus", len(lulus), help=f"Nilai ≥ {cfg.aturan.NILAI_LULUS_MINIMUM} (AD-07)")
    c[1].metric("SKS lulus", kur.total_sks(lulus))
    c[2].metric("IPK", f"{ipk:.2f}" if ipk is not None else "–", help=ket_ipk)
    c[3].metric(f"SKS bernilai {'/'.join(cfg.aturan.NILAI_DIBATASI)}", sks_d,
                help=f"Paling banyak {cfg.aturan.BATAS_PROPORSI_NILAI_DIBATASI:.0%} total beban kredit (AD-07)")
    if gagal:
        st.caption("Belum lulus (akan diprioritaskan bila wajib, AT09): " + ", ".join(sorted(gagal)))
    for x in catatan:
        st.caption(f"⚠️ {x}")

    pilihan_tema = st.session_state.get("f_pilihan_tema", {})
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    if c1.button("Atur MK pilihan per tema", width="stretch"):
        _panel_pilihan_tema(ctx, lulus)
    c2.caption("MK pilihan yang Anda pilih: " + ", ".join(f"Tema {t} → {k}" for t, k in sorted(pilihan_tema.items()))
               if pilihan_tema else "MK pilihan: seluruh tema mengikuti rekomendasi sistem.")

    with st.form("profil"):
        c3, c4 = st.columns(2)
        semester = c3.number_input("Semester berjalan", 1, kur.param_int("AT04", "semester_max"), key="f_sem",
                                   help="Rencana disusun mulai semester ini.")
        ip = c4.number_input("IP semester sebelumnya", 0.0, 4.0, step=0.01, key="f_ip",
                             help="Menentukan pita SKS (09/AT05) dan diasumsikan tetap untuk semester mendatang. "
                                  "Diabaikan di semester paket 1–2.")
        ipk_isian = None
        if ipk is None:
            ipk_isian = st.number_input("IPK (opsional, bila riwayat nilai tidak diisi lengkap)", 0.0, 4.0, value=None,
                                        step=0.01, placeholder="mis. 3.45",
                                        help="Dipakai untuk syarat Tugas Akhir AD-10 (IPK ≥ 2,00) bila IPK tidak dapat "
                                             "dihitung dari riwayat nilai.")
        tema_konv, sks_konv = None, 0
        if jalur != "A":
            with st.container(border=True):
                st.caption(f"Jalur {jalur}: slot konversi merdeka belajar (AT12–AT14). Kosongkan tema bila belum "
                           f"diketahui, planner memakai asumsi yang dicatat di rencana.")
                cc1, cc2 = st.columns([3, 1])
                tema_pilih = cc1.multiselect("Tema pilihan yang dipenuhi lewat konversi", sorted(kur.tema))
                sks_konv = int(cc2.number_input("SKS konversi yang sudah diakui", 0, 60, 0))
                tema_konv = set(tema_pilih) if tema_pilih else None
        c5, c6 = st.columns(2)
        pilihan_mode = ["hybrid", "arm", "keduanya"] if ada_theta else ["arm"]
        mode = c5.radio("Mode", pilihan_mode, horizontal=True,
                        format_func=lambda m: {"keduanya": "Bandingkan keduanya", **LABEL_MODE}[m])
        label = c6.text_input("Label rencana (opsional)", placeholder="mis. simulasi-A",
                              help="Jangan isi nama atau NIM, rencana disimpan tanpa data pribadi.")
        kirim = st.form_submit_button("Bangkitkan Rencana", type="primary")

    if kirim:
        profil = Profile(semester_saat_ini=int(semester), ip_terakhir=float(ip), mk_lulus=lulus,
                         mk_pernah_gagal=gagal, riwayat_nilai=riwayat, jalur=jalur,
                         agama=None if agama == BELUM_DIISI else agama, tema_konversi=tema_konv,
                         sks_konversi=sks_konv, ipk=ipk_isian, pilihan_tema=pilihan_tema)
        mode_list = ["arm", "hybrid"] if mode == "keduanya" else [mode]
        hasil = {}
        for m in mode_list:
            plan = susun_rencana(profil, m, ctx, label=label.strip())
            simpan_rencana(plan, cfg)
            hasil[m] = plan
        st.session_state["rencana_terakhir"] = hasil

    hasil = st.session_state.get("rencana_terakhir")
    if not hasil:
        return
    st.success("Rencana tersimpan dengan kode " + ", ".join(f"`{p.id}`" for p in hasil.values())
               + ", simpan kode ini untuk melihat validasi DPA.")

    def simpan(plan) -> None:   # penyesuaian manual (iterasi 3): perbarui rencana tersimpan & sesi
        simpan_rencana(plan, cfg)
        st.session_state["rencana_terakhir"][plan.mode] = plan

    if len(hasil) == 1:
        plan = next(iter(hasil.values()))
        st.subheader(LABEL_MODE[plan.mode])
        tampilkan_rencana(plan, kur, cfg, kunci="mhs", ctx=ctx, simpan=simpan)
    else:
        t1, t2, t3 = st.tabs(["Perbandingan", LABEL_MODE["hybrid"], LABEL_MODE["arm"]])
        with t1:
            bandingkan(hasil, cfg)
        with t2:
            tampilkan_rencana(hasil["hybrid"], kur, cfg, kunci="mhs-h", ctx=ctx, simpan=simpan)
        with t3:
            tampilkan_rencana(hasil["arm"], kur, cfg, kunci="mhs-a", ctx=ctx, simpan=simpan)


# ---------------------------------------------------------------------
# DPA (UC-04, UC-05)
# ---------------------------------------------------------------------


def halaman_dpa(cfg, ctx: KonteksPlanner) -> None:
    st.header("Tinjauan rencana oleh DPA")
    st.caption("DPA tetap validator akhir: sistem hanya menyusun rancangan awal yang telah diperiksa aturan "
               "akademik (KF-18, KF-19).")
    rencana = daftar_rencana(cfg)
    if not rencana:
        st.info("Belum ada rencana tersimpan. Rencana dibuat dari halaman Mahasiswa.")
        return
    ringkas = pd.DataFrame([{
        "ID": p.id, "Dibuat": p.dibuat.strftime("%d-%m-%Y %H:%M"), "Label": p.label, "Mode": LABEL_SINGKAT[p.mode],
        "Jalur": p.profil.jalur, "Mulai": f"S{p.profil.semester_saat_ini}", "IP": p.profil.ip_terakhir,
        "SKS rencana": p.total_sks_rencana,
        "Lolos aturan": "✅" if p.validasi and p.validasi.lolos else "❌",
        "Perlu verifikasi": len(p.validasi.perlu_verifikasi) if p.validasi else 0,
        "Disesuaikan mahasiswa": f"{len(p.penyesuaian)} pemindahan" if p.disesuaikan else "",
        "Status DPA": LABEL_STATUS_DPA[p.status_dpa],
    } for p in rencana])
    st.dataframe(ringkas, hide_index=True, width="stretch")

    pid = st.selectbox("Pilih rencana", [p.id for p in rencana],
                       format_func=lambda i: next(f"{p.dibuat:%d-%m-%Y %H:%M} · {LABEL_SINGKAT[p.mode]} · "
                                                  f"Jalur {p.profil.jalur} · mulai S{p.profil.semester_saat_ini} · "
                                                  f"{p.label or 'tanpa label'}" for p in rencana if p.id == i))
    plan = muat_rencana(cfg, pid)
    kur = ctx.kurikulum
    prof = plan.profil
    ipk, ket_ipk = hitung_ipk(prof.riwayat_nilai, kur, cfg)
    if ipk is None and prof.ipk is not None:
        ipk, ket_ipk = prof.ipk, "IPK isian mahasiswa / data akademik"
    c = st.columns(5)
    c[0].metric("Jalur", prof.jalur)
    c[1].metric("Semester mulai", prof.semester_saat_ini)
    c[2].metric("IP terakhir", f"{prof.ip_terakhir:.2f}")
    c[3].metric("IPK", f"{ipk:.2f}" if ipk is not None else "–", help=ket_ipk)
    c[4].metric("SKS lulus", kur.total_sks(prof.mk_lulus & set(kur.mk)) + prof.sks_konversi)
    with st.expander(f"MK yang telah diambil ({len(prof.pernah_menempuh)})"):
        st.dataframe(pd.DataFrame([{"Kode": k, "Mata kuliah": kur.mk[k].nama if k in kur else k,
                                    "SKS": kur.sks(k) if k in kur else None,
                                    "Nilai": prof.riwayat_nilai.get(k, "–"),
                                    "Status": "lulus" if k in prof.mk_lulus else "belum lulus"}
                                   for k in sorted(prof.pernah_menempuh)]), hide_index=True, width="stretch")

    with st.form("catatan_dpa"):
        status = st.radio("Validasi akhir DPA", ["disetujui", "perlu_revisi"], horizontal=True,
                          index=1 if plan.status_dpa == "perlu_revisi" else 0, format_func=LABEL_STATUS_DPA.get)
        catatan = st.text_area("Catatan bimbingan", value=plan.catatan_dpa, height=120,
                               placeholder="Masukan untuk mahasiswa, mis. verifikasi syarat PLP, pertimbangkan menunda MK X…")
        if st.form_submit_button("Simpan validasi & catatan", type="primary"):
            plan = perbarui_catatan_dpa(cfg, pid, catatan, status)
            st.success(f"Tersimpan: {LABEL_STATUS_DPA[plan.status_dpa]}.")
    if plan.catatan_dpa_diperbarui:
        st.caption(f"Terakhir ditinjau {plan.catatan_dpa_diperbarui:%d-%m-%Y %H:%M} · status "
                   f"{LABEL_STATUS_DPA[plan.status_dpa]}.")

    st.subheader(LABEL_MODE[plan.mode])
    tampilkan_rencana(plan, kur, cfg, kunci=f"dpa-{pid}")


# ---------------------------------------------------------------------
# Administrator (UC-06..UC-12)
# ---------------------------------------------------------------------


PANDUAN_ADMIN = ROOT / "docs" / "PANDUAN_ADMINISTRATOR.md"


def _tombol_templat(kunci: str, key: str) -> None:
    """Tombol unduh templat CSV kosong (nama kolom saja) + kamus kolomnya."""
    from src.templat import TEMPLAT, csv_kosong, kamus

    t = TEMPLAT[kunci]
    st.download_button(f"Unduh templat {t.berkas}", csv_kosong(kunci), file_name=t.berkas, mime="text/csv",
                       key=f"templat-{key}", help=f"Berisi nama kolom saja. Dipakai di: {t.dipakai}")
    with st.popover(f"Kolom {t.berkas}"):
        st.dataframe(kamus(kunci), hide_index=True, width="stretch")


def tab_mulai(cfg, ctx: KonteksPlanner) -> None:
    """Iterasi 3: titik awal Administrator: daftar periksa dari nol, templat data, cadangan & atur ulang."""
    from src.kesiapan import IKON, OPSIONAL, SELESAI
    from src.templat import BERKAS_XLSX, SHEET_XLSX, TEMPLAT, kamus, xlsx_kosong

    st.markdown("Halaman ini memandu pengisian sistem **dari kondisi kosong** sampai rencana studi divalidasi DPA. "
                "Kerjakan langkah berurutan; status diperbarui otomatis. Penjelasan lengkap setiap langkah ada di "
                "panduan administrator.")
    if PANDUAN_ADMIN.exists():
        st.download_button("Unduh panduan administrator (Markdown)", PANDUAN_ADMIN.read_bytes(),
                           file_name="PANDUAN_ADMINISTRATOR.md", mime="text/markdown")

    langkah = _kesiapan(versi_data())
    _cara_cepat_paket(cfg, kosong=not any(x.status == SELESAI for x in langkah[2:]))
    st.markdown("**Daftar periksa (cara lengkap, langkah demi langkah)**")
    wajib = [x for x in langkah if x.wajib]
    selesai = sum(x.status == SELESAI for x in wajib)
    st.progress(selesai / len(wajib), text=f"Langkah wajib selesai: {selesai} dari {len(wajib)}")
    st.dataframe(pd.DataFrame([{
        "No": x.no, "Status": f"{IKON[x.status]} {x.status}", "Langkah": x.judul + ("" if x.wajib else " (opsional)"),
        "Keterangan": x.keterangan, "Tempat di aplikasi": x.tempat,
    } for x in langkah]), hide_index=True, width="stretch")
    berikut = next((x for x in langkah if x.wajib and x.status != SELESAI), None)
    if berikut:
        st.info(f"**Langkah berikutnya: {berikut.no}. {berikut.judul}.** {berikut.keterangan}. Buka: {berikut.tempat}.")
    else:
        st.success("Semua langkah wajib selesai. Langkah opsional "
                   + ", ".join(str(x.no) for x in langkah if x.status == OPSIONAL) + " dipakai untuk evaluasi RM1."
                   if any(x.status == OPSIONAL for x in langkah) else "Semua langkah selesai.")

    st.divider()
    st.markdown("**Templat data** (berisi nama kolom saja; isi lalu unggah di tempat yang disebutkan)")
    st.caption("CSV boleh disimpan dari Excel (pemisah titik koma dan koma desimal dikenali otomatis). Jangan "
               "menambah kolom NIM atau nama: data mahasiswa harus sudah anonim, kecuali xlsx sistem akademik yang "
               "dianonimkan otomatis saat impor.")
    st.dataframe(pd.DataFrame([{"Berkas": t.berkas, "Isi": t.judul, "Diunggah di": t.dipakai}
                               for t in TEMPLAT.values()] +
                              [{"Berkas": BERKAS_XLSX, "Isi": "Ekspor sistem akademik (2 sheet: KRS, IP/IPK; ber-NIM)",
                                "Diunggah di": "Administrator → Data akademik → Impor data akademik → xlsx akademik"}]),
                 hide_index=True, width="stretch")
    c = st.columns(3)
    for i, kunci in enumerate(TEMPLAT):
        with c[i % 3]:
            _tombol_templat(kunci, f"mulai-{kunci}")
    with c[len(TEMPLAT) % 3]:
        st.download_button("Unduh templat data akademik (xlsx)", xlsx_kosong(), file_name=BERKAS_XLSX,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           help="Nama sheet harus memuat 'Ganjil' atau 'Genap' dan tahun, mis. 'KRS Ganjil 2026'.")
        with st.popover(f"Kolom {BERKAS_XLSX}"):
            for judul, kolom in SHEET_XLSX.items():
                st.markdown(f"Sheet **{judul}**: " + ", ".join(f"`{k}`" for k in kolom))
    st.markdown("**Data acuan untuk mengisi templat**")
    a1, a2 = st.columns(2)
    kur = ctx.kurikulum
    daftar_mk = pd.DataFrame([{"kode_mk": k, "nama_mk": m.nama, "sks": m.sks, "sifat": m.sifat, "tema": m.tema or "",
                               "semester_jalur_A": m.semester_acuan or ""}
                              for k, m in sorted(kur.mk.items(), key=lambda x: (x[1].semester_acuan or 99, x[0]))])
    a1.download_button("Unduh daftar kode MK Kurikulum 2024", daftar_mk.to_csv(index=False).encode("utf-8-sig"),
                       file_name="daftar_mk_kurikulum_2024.csv", mime="text/csv")
    a2.download_button("Unduh formulir penilaian pakar (terisi daftar CPMK)", _formulir_pakar_csv(versi_data()),
                       file_name="formulir_penilaian_pakar.csv", mime="text/csv", key="formulir-pakar-mulai")

    st.divider()
    _cadangan_dan_atur_ulang(cfg)


def _berkas_paket(unggahan) -> tuple[dict[str, pd.DataFrame], dict[str, bytes], list[dict]]:
    """Unggahan (berkas lepas dan/atau zip) → ({jenis: DataFrame} dikenali dari kolomnya, berkas riwayat
    multisemester apa adanya, laporan per berkas)."""
    import zipfile

    from src.berkas import BERKAS_RAW
    from src.paket_uji import JUDUL, kenali

    isi: list[tuple[str, bytes]] = []
    for up in unggahan:
        if up.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(up.getvalue())) as z:
                isi += [(Path(n).name, z.read(n)) for n in z.namelist() if not n.endswith("/")]
        else:
            isi.append((up.name, up.getvalue()))
    berkas, raw, laporan = {}, {}, []
    for nama, data in isi:
        if nama in BERKAS_RAW:      # riwayat multisemester & penanda asal datanya: disalin byte per byte
            raw[nama] = data
            laporan.append({"Berkas": nama, "Dikenali sebagai": "Riwayat KRS multisemester", "Baris": None,
                            "Status": "✅"})
            continue
        if not nama.lower().endswith(".csv"):
            continue
        try:
            df = baca_csv_unggahan(io.BytesIO(data), dtype={"id_anon": str, "kode_mk": str, "penilai": str})
            jenis = kenali(df)
        except Exception as e:   # berkas rusak/bukan CSV
            laporan.append({"Berkas": nama, "Dikenali sebagai": "-", "Baris": 0, "Status": f"❌ tidak terbaca: {e}"})
            continue
        if jenis is None:
            status = "❌ kolom tidak dikenali (bukan berkas paket)"
        elif jenis in berkas:
            status = "⚠️ ganda, diabaikan"
        else:
            berkas[jenis], status = df, "✅"
        laporan.append({"Berkas": nama, "Dikenali sebagai": JUDUL.get(jenis, "-"), "Baris": len(df), "Status": status})
    return berkas, raw, laporan


def _cara_cepat_paket(cfg, kosong: bool) -> None:
    """Iterasi 3: dari kosong menjadi siap pakai dengan paket data uji (3 CSV) dalam satu kali proses."""
    from src.paket_uji import BERKAS, JUDUL, siapkan_dari_paket

    with st.expander("Cara cepat: siapkan sistem dari paket data uji (± 1 menit, tanpa API berbayar)", expanded=kosong):
        st.markdown(
            "1. Pastikan baris **1. Konfigurasi .env / app settings** pada daftar periksa tidak berstatus ⬜ "
            "(kata sandi peran terisi). Kunci API LLM tidak diperlukan.\n"
            "2. Pilih **satu berkas zip paket data uji** (atau seluruh isinya sekaligus: `"
            + "`, `".join(BERKAS.values()) + "`, dan berkas riwayat multisemester). Jenis berkas dikenali "
            "otomatis.\n"
            "3. Klik **Siapkan sistem dari paket data uji**. Sistem mengimpor data akademik, memasang riwayat "
            "multisemester, memulihkan rating Bloom LLM, lalu menjalankan preprocessing, ARM, CL & θ, evaluasi, "
            "simulasi, dan validasi KRS.\n"
            "4. Setelah daftar periksa menunjukkan langkah 3 sampai 8 dan 11 ✅, lanjutkan uji di peran Mahasiswa "
            "dan DPA (skenario U-09 sampai U-14 di panduan administrator).")
        if not kosong:
            st.caption("Sistem sudah berisi data. Untuk pengujian dari nol, atur ulang dulu (bagian Cadangan & atur "
                       "ulang di bawah). Periode KRS yang sama di paket akan menggantikan data saat ini.")
        unggahan = st.file_uploader("Berkas paket data uji", type=["csv", "zip", "txt", "json"],
                                    accept_multiple_files=True, key="up_paket_uji")
        berkas, raw, laporan = _berkas_paket(unggahan or [])
        if laporan:
            st.dataframe(pd.DataFrame(laporan), hide_index=True, width="stretch")
            kurang = [JUDUL[k] for k in BERKAS if k not in berkas]
            if kurang:
                st.caption("Belum ada: " + ", ".join(kurang) + (". KRS akademik wajib; tanpa rating LLM, skor CL "
                                                                "memakai heuristik." if "krs" in berkas else "."))
        if st.button("Siapkan sistem dari paket data uji", type="primary", disabled="krs" not in berkas):
            bar = st.progress(0.0, text="Memulai…")
            try:
                r = siapkan_dari_paket(cfg, berkas, raw, progress=lambda i, n, t: bar.progress(i / n, text=t))
            except (ValueError, FileExistsError) as e:
                st.error(f"Ditolak: {e}")
            else:
                st.cache_resource.clear()
                st.session_state["pesan_paket"] = (f"Sistem siap: θ = {r['theta']}, {r['n_aturan']} aturan asosiasi. "
                                                   f"Lanjutkan ke peran Mahasiswa.")
                st.rerun()
        if pesan := st.session_state.pop("pesan_paket", None):
            st.success(pesan)


def _cadangan_dan_atur_ulang(cfg) -> None:
    from src.kesiapan import atur_ulang
    from src.llm.beban import impor_rating_llm, jalankan_cl
    from src.templat import baca_csv_unggahan

    st.markdown("**Cadangan & atur ulang**")
    cadangan = _cadangan_rating(versi_data())
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Rating Bloom LLM dihasilkan lewat API berbayar. Unduh cadangannya sebelum mengatur ulang agar "
                   "dapat dipulihkan tanpa memanggil API lagi (model, versi prompt, dan waktu penilaian ikut tersimpan).")
        if len(cadangan):
            st.download_button(f"Unduh cadangan rating LLM ({len(cadangan):,} baris)",
                               cadangan.to_csv(index=False).encode("utf-8-sig"), file_name="cadangan_rating_llm.csv",
                               mime="text/csv")
        else:
            st.caption("Belum ada rating LLM tersimpan.")
        try:
            st.download_button("Unduh paket data uji (zip)", _zip_paket_data(versi_data()),
                               file_name="paket_data_uji.zip", mime="application/zip",
                               help="KRS akademik anonim, IP/IPK, rating LLM, dan riwayat multisemester saat ini, untuk "
                                    "menyiapkan sistem kosong lewat Cara cepat. Berisi data akademik anonim: serahkan "
                                    "langsung, jangan diunggah ke repositori publik.")
        except ValueError:
            st.caption("Paket data uji tersedia setelah data akademik dan rating LLM ada di basis data.")
        up = st.file_uploader("Berkas cadangan rating LLM (CSV)", type=["csv"], key="up_rating_llm")
        if up is not None and st.button("Pulihkan rating LLM", type="primary"):
            try:
                r = impor_rating_llm(cfg, baca_csv_unggahan(up, dtype={"penilai": str}))
                jalankan_cl(cfg)
                hitung_dan_simpan_theta(cfg)
            except ValueError as e:
                st.error(f"Ditolak: {e}")
            else:
                st.success(f"{r['n_baris']:,} penilaian ({r['n_mk']} MK, {r['n_pengulangan']} pengulangan; "
                           f"{', '.join(r['run'])}) dipulihkan. CL dan θ dihitung ulang.")
    with c2:
        st.caption("Atur ulang mengosongkan data akademik, riwayat, transaksi, aturan, skor CL, θ, rencana, override, "
                   "dan parameter UI, sehingga sistem dapat diuji dari nol. Dataset kurikulum, log proses, dan log "
                   "akses tidak dihapus. Tindakan ini tidak dapat dibatalkan.")
        tetap = st.checkbox("Pertahankan rating Bloom LLM (tidak perlu memanggil API berbayar lagi)", value=True)
        konfirmasi = st.text_input("Ketik ATUR ULANG untuk konfirmasi", key="konfirmasi_atur_ulang")
        if st.button("Atur ulang sistem ke kondisi awal", disabled=konfirmasi.strip() != "ATUR ULANG"):
            r = atur_ulang(cfg, pertahankan_rating_llm=tetap)
            for k in ("rencana_terakhir", "skenario_hasil", "konfirmasi_atur_ulang"):
                st.session_state.pop(k, None)
            st.cache_resource.clear()
            st.session_state["pesan_atur_ulang"] = (
                f"Sistem diatur ulang: {sum(r['baris_dihapus'].values()):,} baris dari "
                f"{len(r['baris_dihapus'])} tabel dan {len(r['berkas_dihapus'])} berkas dihapus.")
            st.rerun()
    if pesan := st.session_state.pop("pesan_atur_ulang", None):
        st.success(pesan)


def tab_kurikulum(cfg, ctx: KonteksPlanner) -> None:
    kur = ctx.kurikulum
    st.markdown(f"Dataset Kurikulum 2024 di `{cfg.paths.kurikulum}` (hasil `scripts/kurikulum/build_dataset.py`, "
                f"hanya dibaca, koreksi data dilakukan di skrip build + `19_catatan_inkonsistensi.csv`).")
    c = st.columns(5)
    c[0].metric("Mata kuliah", len(kur.mk))
    c[1].metric("Prasyarat", len(kur.prasyarat))
    c[2].metric("Tema pilihan", len(kur.tema))
    c[3].metric("SKS Jalur A", kur.total_sks_jalur("A"))
    c[4].metric("SKS Jalur B–F", kur.total_sks_jalur("B"))
    berkas = sorted(p.name for p in cfg.kurikulum_dir.glob("*.csv"))
    nama = st.selectbox("Berkas", berkas, index=berkas.index(FILE_KURIKULUM["mk"]) if FILE_KURIKULUM["mk"] in berkas else 0)
    df = baca_csv_kurikulum(cfg, nama)
    st.caption(f"{len(df):,} baris · {len(df.columns)} kolom")
    st.dataframe(df, hide_index=True, width="stretch", height=320)
    _override_cpmk(cfg, ctx)
    _override_prasyarat(cfg, ctx)


def _override_cpmk(cfg, ctx: KonteksPlanner) -> None:
    """CPMK tidak lengkap/ambigu ditandai lalu dapat dilengkapi administrator (KF-03)."""
    from src.data_loader import load_konteks
    from src.override import daftar_override_cpmk, hapus_override_cpmk, simpan_override_cpmk

    st.divider()
    st.markdown("**CPMK yang perlu ditinjau**: KKO ambigu (`kko_ambigu = Y`) atau rumusan sangat pendek; "
                "lengkapi lewat override (dataset tidak diubah, teks asli tetap tersimpan).")
    cpmk = load_konteks(cfg, "cpmk")
    tinjau = cpmk[(cpmk["kko_ambigu"] == "Y") | (cpmk["teks_dataset"].str.len() < 40) | cpmk["override"]]
    st.caption(f"{len(tinjau)} dari {len(cpmk)} CPMK · {int(cpmk['override'].sum())} sudah di-override")
    st.dataframe(tinjau[["id_cpmk", "kode_mk", "teks", "bloom_maks_heuristik", "kko_ambigu", "override"]],
                 hide_index=True, width="stretch", height=220)
    with st.form("form_override_cpmk"):
        id_cpmk = st.selectbox("CPMK", list(tinjau["id_cpmk"]) or list(cpmk["id_cpmk"]))
        teks_lama = cpmk.set_index("id_cpmk").loc[id_cpmk, "teks_dataset"]
        st.caption(f"Rumusan dataset: {teks_lama}")
        teks = st.text_area("Rumusan CPMK yang dipakai sistem", height=80)
        alasan = st.text_input("Alasan perubahan (wajib)")
        if st.form_submit_button("Simpan override CPMK"):
            try:
                simpan_override_cpmk(cfg, id_cpmk, teks, alasan, cpmk)
                st.success("Override CPMK tersimpan. Jalankan ulang penilaian Bloom untuk MK ini.")
            except ValueError as e:
                st.error(str(e))
    ovr = daftar_override_cpmk(cfg)
    if len(ovr):
        st.dataframe(ovr, hide_index=True, width="stretch")
        hapus = st.selectbox("Hapus override CPMK", ["-"] + list(ovr["id_cpmk"]), key="hapus_ovr_cpmk")
        if hapus != "-" and st.button("Hapus override CPMK terpilih"):
            hapus_override_cpmk(cfg, hapus)
            st.rerun()


def _override_prasyarat(cfg, ctx: KonteksPlanner) -> None:
    """KF-02: tambah/hapus relasi prasyarat "nilai MK" tanpa mengubah dataset (graf harus tetap asiklik)."""
    from src.override import daftar_override_prasyarat, hapus_override_prasyarat, simpan_override_prasyarat

    kur = ctx.kurikulum
    st.divider()
    st.markdown("**Override prasyarat** (KF-02), tervalidasi: kode dikenal, relasi yang dihapus ada, tanpa siklus.")
    opsi = sorted(kur.kode)
    with st.form("form_override_prasyarat"):
        c1, c2, c3 = st.columns([2, 2, 1])
        kode = c1.selectbox("MK", opsi, format_func=kur.label)
        syarat = c2.selectbox("MK prasyarat", opsi, format_func=kur.label)
        aksi = c3.radio("Aksi", ["tambah", "hapus"])
        alasan = st.text_input("Alasan (wajib)", key="alasan_ovr_pr")
        if st.form_submit_button("Simpan override prasyarat"):
            try:
                simpan_override_prasyarat(cfg, kode, syarat, aksi, alasan)
                st.success("Override prasyarat tersimpan.")
            except ValueError as e:
                st.error(str(e))
    ovr = daftar_override_prasyarat(cfg)
    if len(ovr):
        st.dataframe(ovr, hide_index=True, width="stretch")
        pilih = st.selectbox("Hapus override prasyarat", ["-"] + [f"{r.kode_mk}|{r.kode_mk_syarat}"
                                                                  for r in ovr.itertuples()], key="hapus_ovr_pr")
        if pilih != "-" and st.button("Hapus override prasyarat terpilih"):
            hapus_override_prasyarat(cfg, *pilih.split("|"))
            st.rerun()


def tab_data_akademik(cfg) -> None:
    """UC-06: data akademik (KRS + IP/IPK) dan riwayat multisemester."""
    from src.akademik import baca_xlsx, impor, muat_ip, muat_krs, ringkasan, validasi

    st.markdown("**KRS & IP/IPK akademik**: diimpor ke basis data tanpa NIM (dianonimkan dengan `ANON_SALT`). "
                "Berkas mentah tidak pernah disimpan.")
    r = ringkasan(cfg)
    if r:
        c = st.columns(4)
        c[0].metric("Periode KRS", ", ".join(r["periode_krs"]))
        c[1].metric("Mahasiswa (KRS)", r["n_mahasiswa_krs"])
        c[2].metric("Mahasiswa (IP)", r["n_mahasiswa_ip"], help=", ".join(r["periode_ip"]))
        c[3].metric("Kode MK tak dikenal", len(r["kode_mk_tidak_dikenal"]))
        st.caption("Mahasiswa per semester: " + ", ".join(f"S{s}: {n}" for s, n in r["mahasiswa_per_semester"].items())
                   + f" · diimpor {r.get('diimpor', '–')} dari {r.get('asal', '–')}")
        with st.expander("Pratinjau data anonim"):
            st.dataframe(muat_krs(cfg).head(200), hide_index=True, width="stretch", height=220)
            st.dataframe(muat_ip(cfg).head(200), hide_index=True, width="stretch", height=220)
    else:
        st.info("Data akademik belum diimpor.")
    with st.expander("Impor data akademik"):
        st.caption("xlsx dari akademik (sheet `KRS <Ganjil|Genap> <tahun>` + `IP dan IPK <Genap|Ganjil> <tahun>`) "
                   "atau CSV anonim (`histori_krs.csv` + `ip_semester.csv`, format KAMUS_DATA). Periode yang sama "
                   "diganti.")
        jenis = st.radio("Format", ["xlsx akademik", "CSV anonim"], horizontal=True, key="fmt_akademik")
        if jenis == "xlsx akademik":
            from src.templat import BERKAS_XLSX, xlsx_kosong

            st.download_button("Unduh templat data akademik (xlsx)", xlsx_kosong(), file_name=BERKAS_XLSX,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="templat-xlsx-akademik")
        else:
            t1, t2 = st.columns(2)
            with t1:
                _tombol_templat("krs", "akademik-krs")
            with t2:
                _tombol_templat("ip", "akademik-ip")
        try:
            if jenis == "xlsx akademik":
                up = st.file_uploader("Berkas xlsx", type=["xlsx"], key="up_xlsx")
                data = baca_xlsx(up.getvalue()) if up is not None else None
                asal = f"xlsx {up.name}" if up is not None else ""
            else:
                c1, c2 = st.columns(2)
                up_krs = c1.file_uploader("histori_krs.csv", type=["csv"], key="up_krs_asli")
                up_ip = c2.file_uploader("ip_semester.csv (opsional)", type=["csv"], key="up_ip_asli")
                data = None
                if up_krs is not None:
                    data = (baca_csv_unggahan(up_krs, dtype={"id_anon": str, "kode_mk": str}),
                            baca_csv_unggahan(up_ip, dtype={"id_anon": str}) if up_ip is not None else pd.DataFrame())
                asal = f"CSV {up_krs.name}" if up_krs is not None else ""
            if data is not None:
                _, _, lap = validasi(*data, cfg)
                st.success(f"Valid: {lap['n_baris_krs']:,} baris KRS ({lap['n_mahasiswa_krs']} mahasiswa, "
                           f"{', '.join(lap['periode_krs'])}); {lap['n_mahasiswa_ip']} mahasiswa ber-IP.")
                if lap["kode_mk_tidak_dikenal"]:
                    st.warning(f"Kode MK tidak dikenal Kurikulum 2024: {', '.join(lap['kode_mk_tidak_dikenal'])}")
                if st.button("Simpan ke basis data", type="primary", key="simpan_akademik"):
                    impor(cfg, *data, asal=asal)
                    st.success("Tersimpan. Jalankan ulang Preprocessing & ARM (atau pipeline penuh).")
        except Exception as e:
            st.error(f"Ditolak: {e}")

    st.divider()
    status = status_data_krs(cfg)
    st.markdown(f"**Riwayat KRS multisemester** (`{cfg.paths.raw}/histori_krs.csv`), "
                f"{'belum tersedia' if status == 'belum_tersedia' else 'tersedia'}. Dipakai untuk semester yang "
                f"belum ada di KRS akademik (ARM, θ) dan simulasi UC-12.")
    if status in ("belum_tersedia", "sintetis"):
        if st.button("Bangkitkan riwayat multisemester"):
            from src.sintetis import generate

            with st.spinner("Membangkitkan riwayat…"):
                generate(cfg)
            st.success("Riwayat dibangkitkan. Jalankan ulang Preprocessing & ARM.")
    path = cfg.raw_dir / "histori_krs.csv"
    with st.expander("Unggah riwayat alumni (histori_krs.csv, UC-06)"):
        st.caption(f"Kolom wajib: {', '.join(KOLOM_KRS_WAJIB)} + salah satu kolom ID {cfg.preprocessing.kolom_id}. "
                   f"Kode Kurikulum 2020 dinormalisasi lewat `11_penyetaraan`. Menggantikan riwayat yang ada.")
        _tombol_templat("histori", "histori")
        up = st.file_uploader("CSV histori KRS", type=["csv"], key="up-krs")
        if up is not None:
            try:
                baru = baca_csv_unggahan(up, dtype=str)
                kurang = [k for k in KOLOM_KRS_WAJIB if k not in baru.columns]
                if kurang or not any(k in baru.columns for k in cfg.preprocessing.kolom_id):
                    raise ValueError(f"kolom kurang: {kurang or cfg.preprocessing.kolom_id}")
            except Exception as e:
                st.error(f"File ditolak: {e}")
            else:
                st.success(f"Valid: {len(baru):,} baris.")
                if st.button("Simpan sebagai histori_krs.csv", type="primary"):
                    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
                    baru.to_csv(path, index=False)
                    for f in ("SUMBER_DATA.txt", "kalibrasi_sintetis.json"):
                        if (cfg.raw_dir / f).exists():
                            (cfg.raw_dir / f).unlink()
                    simpan_berkas(cfg)
                    st.success("Tersimpan. Jalankan preprocessing & ARM.")


def tab_preprocessing(cfg) -> None:
    st.markdown("KRS akademik (semester ≤ batas masa studi) + riwayat multisemester (anonimisasi → filter masa "
                "studi → status lulus → normalisasi kode 2020 → 2024) → transaksi `kode@S` (MK diambil) → split "
                "latih:uji per mahasiswa.")
    ada_krs = any(sumber_data(cfg).values())
    if st.button("Jalankan preprocessing", type="primary", disabled=not ada_krs):
        from src.preprocessing import jalankan_preprocessing

        with st.spinner("Memproses…"):
            jalankan_preprocessing(cfg)
        st.success("Selesai. Jalankan ARM agar aturan konsisten dengan data terbaru.")
    if not ada_krs:
        st.info("Data KRS belum tersedia (impor data akademik atau sediakan riwayat).")
    r = db.baca_ringkasan(cfg, "preprocessing")
    if not r or "transaksi_per_semester_sumber" not in r:
        return
    c = st.columns(4)
    c[0].metric("Transaksi KRS akademik", r["n_transaksi_asli"],
                help=f"mahasiswa latih {r['n_mahasiswa_asli_latih']} · uji {r['n_mahasiswa_asli_uji']}")
    c[1].metric("Transaksi riwayat", r["n_transaksi_riwayat"],
                help=f"alumni latih {r['n_alumni_latih']} · uji {r['n_alumni_uji']}")
    c[2].metric("Transaksi latih", r["n_transaksi_latih"])
    c[3].metric("Transaksi uji", r["n_transaksi_uji"])
    st.markdown("**Transaksi per semester**")
    st.dataframe(pd.DataFrame([{"Semester": s, "Transaksi": sum(d.values())}
                               for s, d in r["transaksi_per_semester_sumber"].items()]), hide_index=True)
    if "n_alumni_awal" in r:
        c = st.columns(4)
        c[0].metric("Alumni riwayat", r["n_alumni_awal"])
        c[1].metric("Alumni ≤ batas masa studi", r["n_alumni_masa_studi_ok"],
                    delta=f"-{r['n_alumni_dibuang_masa_studi']} dibuang", delta_color="off")
        c[2].metric("Baris terpetakan", f"{r['proporsi_baris_terpetakan']:.1%}")
        c[3].metric("Baris berubah SKS", r["n_baris_berubah_sks"])
        if r.get("mk_dihapus") or r.get("mk_tidak_dikenal"):
            st.markdown(f"**MK tanpa padanan** (kebijakan MK dihapus: `{r['kebijakan_mk_dihapus']}`)")
            st.dataframe(pd.DataFrame([{"Kode": k, "Jenis": "dihapus", "Baris": n} for k, n in r["mk_dihapus"].items()]
                                      + [{"Kode": k, "Jenis": "tidak dikenal", "Baris": n}
                                         for k, n in r["mk_tidak_dikenal"].items()]), hide_index=True)
    st.caption(f"run_id {r.get('run_id')}")


def _label_itemset(teks: str, ctx: KonteksPlanner) -> str:
    return ", ".join(f"{ctx.kurikulum.mk[k.split('@')[0]].nama if k.split('@')[0] in ctx.kurikulum else k}"
                     f" ({k.split('@')[1]})" for k in str(teks).split(";"))


def tab_arm(cfg, ctx: KonteksPlanner) -> None:
    a = cfg.arm
    ada_latih = ada_transaksi(cfg, "latih")
    st.markdown("FP-Growth pada transaksi **latih** (keranjang MK yang diambil per mahasiswa-semester). Ambang "
                "eksak (jumlah minimum ⌈min_support × n⌉); aturan berkonsekuen MK wajib dibuang (§6). Per "
                f"semester: KRS akademik bila transaksinya ≥ {a.min_transaksi_asli}, selain itu riwayat multisemester.")
    c = st.columns(4)
    sup = c[0].number_input("min_support", 0.01, 1.0, a.min_support, 0.01)
    conf = c[1].number_input("min_confidence", 0.05, 1.0, a.min_confidence, 0.05)
    lift = c[2].number_input("min_lift", 0.0, 10.0, a.min_lift, 0.1)
    max_len = c[3].number_input("max_len", 2, 6, a.max_len or 3)
    simpan_default = st.checkbox("Simpan parameter ini sebagai default (config.override.yaml)")
    if st.button("Jalankan FP-Growth", type="primary", disabled=not ada_latih):
        from src.arm import jalankan_arm

        with st.spinner("Menambang aturan…"):
            r = jalankan_arm(cfg, sup, conf, lift, int(max_len))
        if simpan_default:
            save_override({"arm": {"min_support": sup, "min_confidence": conf, "min_lift": lift,
                                   "max_len": int(max_len)}})
            simpan_berkas(cfg)
        st.success(f"{r['n_aturan']} aturan dari {r['n_transaksi_latih']} transaksi latih "
                   f"({r['n_aturan_konsekuen_wajib_dibuang']} aturan berkonsekuen MK wajib dibuang).")
        if not r["n_aturan"]:   # UC-08 alur alternatif A2
            st.warning("Tidak ada aturan yang memenuhi ambang. Turunkan min_support/min_confidence atau lihat "
                       "analisis sensitivitas di bawah.")
    if not ada_latih:
        st.info("Transaksi latih belum ada, butuh data KRS alumni + preprocessing.")
    if ctx.arm.catatan:
        st.caption(f"Status model ARM untuk planner: {ctx.arm.catatan}")
    r = db.baca_ringkasan(cfg, "arm")
    aturan = load_aturan(cfg)
    if r and "per_kelompok" in r:
        st.caption(f"Run terakhir `{r['run_id']}` · parameter {r['parameter']} · rata-rata confidence "
                   f"{r['rata_confidence']} · rata-rata lift {r['rata_lift']}")
        st.dataframe(pd.DataFrame(r["per_kelompok"]).T.drop(columns="sumber", errors="ignore")
                     .rename_axis("Semester").reset_index(),
                     hide_index=True, width="stretch")
    if aturan is not None and len(aturan):
        st.markdown("**20 aturan teratas berdasarkan lift**")
        top = aturan.head(20).copy()
        top["Jika mengambil"] = top["anteseden"].map(lambda t: _label_itemset(t, ctx))
        top["maka juga mengambil"] = top["konsekuen"].map(lambda t: _label_itemset(t, ctx))
        st.dataframe(top[["Jika mengambil", "maka juga mengambil", "support", "confidence", "lift"]],
                     hide_index=True, width="stretch")
    st.divider()
    st.markdown("**Analisis sensitivitas**: grid support × confidence (dari config)")
    if st.button("Jalankan analisis sensitivitas", disabled=not ada_latih):
        from src.arm import jalankan_sensitivitas

        with st.spinner("Menjalankan grid…"):
            jalankan_sensitivitas(cfg)
    sens = db.baca_hasil(cfg, "sensitivitas_arm")
    if sens is not None:
        st.dataframe(sens, hide_index=True, width="stretch")


def tab_beban(cfg, ctx: KonteksPlanner) -> None:
    from src.llm.beban import konsistensi_pengulangan, muat_rating, tabel_cl

    st.markdown("Beban kognitif = level Bloom per CPMK (instrumen pakar DPA: C1–C6, **A** = afektif). "
                "LLM-as-judge **berbatch**: setiap pengulangan = satu permintaan berisi seluruh MK (JSON schema ketat), "
                f"diulang {cfg.llm.n_pengulangan}× lalu dipisah per MK. Yang dikirim hanya nama, deskripsi, CPMK, dan "
                "Sub-CPMK berlabel lokal (tanpa kode MK, SKS, semester, atau skor lain).")
    rating = muat_rating(cfg, "llm")
    if rating is not None:
        terbaru = rating.sort_values("timestamp").iloc[-1]
        st.success(f"Hasil LLM tersimpan dipakai: **{terbaru['model']} · {terbaru['versi_prompt']}**. "
                   f"{rating['kode_mk'].nunique()} MK, {rating['id_cpmk'].nunique()} CPMK, "
                   f"{rating['penilai'].nunique()} pengulangan, terakhir {terbaru['timestamp']}. "
                   "Tidak perlu memanggil API lagi kecuali kurikulum/CPMK berubah.")
    if st.button("Hitung ulang CL & θ dari hasil tersimpan", help="Tanpa memanggil API: konsensus Bloom → CL per MK → θ."):
        from src.llm.beban import jalankan_cl

        jalankan_cl(cfg)
        hitung_dan_simpan_theta(cfg)
        st.success("CL dan θ dihitung ulang dari rating tersimpan.")

    with st.expander("Jalankan penilaian Bloom baru (memanggil API berbayar)"):
        c = st.columns(3)
        provider = c[0].selectbox("Provider", ["openai", "anthropic", "gemini", "mock"],
                                  index=["openai", "anthropic", "gemini", "mock"].index(cfg.llm.provider))
        model_awal = cfg.llm.model if provider == cfg.llm.provider else ""
        model = c[1].text_input("Model", value=model_awal, placeholder="kosong = default provider")
        lanjutkan = c[2].checkbox("Lanjutkan (lewati yang sudah ada)", value=True,
                                  help="Lewati (MK, pengulangan) yang sudah berhasil untuk model & versi prompt yang sama.")
        ukuran = "seluruh MK" if not cfg.llm.mk_per_batch else f"{cfg.llm.mk_per_batch} MK"
        st.caption(f"Prompt {cfg.llm.versi_prompt_bloom}: {cfg.llm.n_pengulangan} pengulangan × 1 permintaan "
                   f"({ukuran} per batch), {cfg.llm.konkurensi} paralel. MK yang tidak lengkap dikirim ulang sendiri; "
                   "keluaran terpotong → batch dibelah dua.")
        setuju = provider == "mock" or st.checkbox("Saya paham ini memanggil API berbayar dengan kunci di `.env`")
        if st.button("Jalankan penilaian Bloom", type="primary", disabled=not setuju):
            from src.export import ekspor_evaluasi
            from src.llm.beban import jalankan_cl, jalankan_penilaian_bloom
            from src.llm.client import LLMKonfigurasiError

            bar = st.progress(0.0, text="Memulai…")
            try:
                r = jalankan_penilaian_bloom(cfg, provider=provider, model=model or None, lanjutkan=lanjutkan,
                                             progress=lambda i, n, k: bar.progress(i / n, text=f"[{i}/{n}] {k}"))
                jalankan_cl(cfg)
                hitung_dan_simpan_theta(cfg)
                ekspor_evaluasi(cfg)
                bar.progress(1.0, text="Selesai")
                st.success(f"{r['n_permintaan_api']} permintaan API · {r['n_mk_dinilai']} MK×pengulangan dinilai · "
                           f"{r['n_perlu_tinjau_manual']} perlu tinjau manual · {r['versi_prompt']} · token "
                           f"{r['token_masuk']:,} masuk / {r['token_keluar']:,} keluar. CL, θ, dan evaluasi dihitung ulang.")
            except LLMKonfigurasiError as e:
                st.error(str(e))

    tabel, info = tabel_cl(cfg, ctx.kurikulum)
    st.markdown(f"**CL per MK**: {info['sumber']}" + (f" · {info['catatan']}" if info.get("catatan") else ""))
    tanpa = tabel[tabel["cl"].isna()]
    if len(tanpa):
        st.caption("Tanpa CL (tidak memiliki CPMK di dokumen; tidak pernah direkomendasikan): "
                   + ", ".join(tanpa["kode_mk"]))
    st.dataframe(tabel, hide_index=True, width="stretch", height=300,
                 column_config={"cl": st.column_config.ProgressColumn("CL", min_value=1, max_value=6, format="%.2f")})
    sumber_rating = info.get("sumber_rating")
    r_aktif = muat_rating(cfg, sumber_rating) if sumber_rating in ("llm", "llm_mock") else None
    if r_aktif is not None:
        df, rk = konsistensi_pengulangan(r_aktif)
        st.markdown("**Konsistensi antarpengulangan (batch) per CPMK**")
        k = st.columns(4)
        k[0].metric("CPMK", rk.get("n_cpmk", 0))
        k[1].metric("Sepakat penuh", f"{(rk.get('proporsi_sepakat_penuh') or 0):.0%}",
                    help="Label Bloom identik di semua pengulangan")
        k[2].metric("Selisih ≤ 1 level", f"{(rk.get('proporsi_rentang_le1') or 0):.0%}")
        k[3].metric("Campur A/C", rk.get("n_campur_afektif", 0), help="Sebagian pengulangan afektif, sebagian kognitif")
        hanya = st.checkbox("Tampilkan hanya CPMK yang tidak sepakat", value=True)
        st.dataframe(df[~df["sepakat_penuh"]] if hanya else df, hide_index=True, width="stretch", height=260)
        st.download_button("Unduh konsistensi per CPMK (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="bloom_konsistensi.csv", mime="text/csv")
    batch = db.baca(cfg, "SELECT * FROM batch_llm ORDER BY timestamp DESC, id_batch")
    if len(batch):
        with st.expander(f"Riwayat permintaan batch ({len(batch)})"):
            st.dataframe(batch, hide_index=True, width="stretch")
    _impor_pakar(cfg)
    st.markdown("**RM1: kesesuaian skor beban kognitif LLM vs pakar (tingkat mata kuliah)**")
    if st.button("Hitung ulang evaluasi"):
        from src.export import ekspor_evaluasi

        ekspor_evaluasi(cfg)
    ev = db.baca_ringkasan(cfg, "evaluasi")
    if ev:
        _ringkasan_rm1(ev.get("rm1") or {})
        with st.expander("Pelengkap tingkat CPMK: weighted Cohen's kappa & akurasi ±1"):
            if ev.get("catatan"):
                st.caption(ev["catatan"])
            baris = [{"Pasangan": p, "Subset": s, **v} for p, isi in ev["kesepakatan"].items()
                     for s, v in isi.items()]
            if baris:
                st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch")
    c = st.columns(3)
    c[0].download_button("Unduh formulir penilaian pakar (terisi daftar CPMK)", _formulir_pakar_csv(versi_data()),
                         file_name="formulir_penilaian_pakar.csv", mime="text/csv", key="formulir-pakar-beban",
                         help="Isi kolom kode_pakar dan bloom, lalu impor (UC-10).")
    skor = db.baca_hasil(cfg, "rm1_skor_mk")
    if skor is not None:
        c[2].download_button("Unduh rm1_skor_mk.csv (ICC di JASP)", skor.to_csv(index=False).encode("utf-8"),
                             file_name="rm1_skor_mk.csv", mime="text/csv")
    banding = db.baca_hasil(cfg, "bloom_perbandingan")
    if banding is not None:
        c[1].download_button("Unduh bloom_perbandingan.csv (untuk JASP)", banding.to_csv(index=False).encode("utf-8"),
                             file_name="bloom_perbandingan.csv", mime="text/csv")


def _impor_pakar(cfg) -> None:
    """UC-10: impor formulir penilaian pakar (KD-07) ke basis data."""
    from src.llm.beban import impor_pakar

    with st.expander("Impor penilaian pakar (UC-10)"):
        st.caption("Format: `id_cpmk, kode_mk, kode_pakar, bloom, catatan` (dari templat). Data pakar dengan "
                   "kode yang sama diganti; bloom kosong = belum dinilai.")
        _tombol_templat("pakar", "pakar")
        up = st.file_uploader("CSV penilaian pakar", type=["csv"], key="up_pakar")
        if up is not None and st.button("Impor penilaian pakar", type="primary"):
            try:
                r = impor_pakar(cfg, baca_csv_unggahan(up, dtype=str, keep_default_na=False))
            except ValueError as e:
                st.error(f"Ditolak: {e}")
            else:
                from src.export import ekspor_evaluasi

                ekspor_evaluasi(cfg)
                st.success(f"{r['n_dinilai']} penilaian dari pakar {', '.join(r['pakar'])} diimpor; evaluasi diperbarui.")


def _ringkasan_rm1(rm1: dict) -> None:
    if rm1.get("peringatan"):
        st.warning(rm1["peringatan"])
    if rm1.get("catatan"):
        st.caption(rm1["catatan"])

    def baris(nama, r):
        if not r or "ICC_A1" not in r:
            return None
        ci = r.get("ICC_A1_ci95") or [None, None]
        return {"Analisis": nama, "n MK": r["n_objek"], "penilai": r["n_penilai"], "ICC(A,1)": r["ICC_A1"],
                "CI95 bawah": ci[0], "CI95 atas": ci[1], "ICC(C,1)": r.get("ICC_C1"), "Kategori": r.get("kategori_A1"),
                "σ² objek": r.get("var_objek"), "σ² penilai": r.get("var_penilai"),
                "σ² galat": r.get("var_interaksi_galat"), "MAE": r.get("MAE")}
    tabel = [b for b in (baris("Utama: LLM vs rata-rata pakar", rm1.get("utama_llm_vs_rata_pakar")),
                         baris("Pembanding: antarpakar", rm1.get("antarpakar")),
                         baris("Konsistensi: antarpengulangan LLM", rm1.get("antarpengulangan_llm")),
                         baris("Baseline: heuristik KKO vs pakar", rm1.get("baseline_heuristik_vs_rata_pakar")))
             if b]
    if tabel:
        st.dataframe(pd.DataFrame(tabel), hide_index=True, width="stretch")
    utama = rm1.get("utama_llm_vs_rata_pakar") or {}
    ba, t = utama.get("bland_altman") or {}, utama.get("uji_t_berpasangan") or {}
    if "bias" in ba:
        st.caption(f"Bland-Altman: bias {ba['bias']}, batas kesepakatan [{ba['loa_bawah']}; {ba['loa_atas']}] · "
                   f"uji t berpasangan: t = {t.get('t')}, p = {t.get('p')}, Shapiro-Wilk p = {t.get('shapiro_p')}")
    st.caption(f"Sumber LLM: {rm1.get('sumber_llm')} · agregasi skor MK: {rm1.get('metode_agregasi')} · "
               f"pratinjau, hasil resmi dihitung di JASP dari rm1_skor_mk.csv.")


def tab_parameter(cfg, ctx: KonteksPlanner) -> None:
    st.markdown("Perubahan disimpan ke `config.override.yaml` (di-merge di atas `config.yaml`). Angka aturan "
                "(145/160 SKS, pita SKS, masa studi) dibaca dari dataset; di sini hanya tafsiran & parameter.")
    info = ctx.info_theta
    c = st.columns(3)
    c[0].metric("θ berlaku", f"{ctx.theta:.2f}" if ctx.theta is not None else "–")
    c[1].metric("Dikalibrasi data KRS", "ya" if info.get("dikalibrasi_data") else "belum")
    c[2].metric("Metode beban", cfg.beban.metode)
    st.caption(f"Sumber θ: {info.get('sumber', '–')}")
    if info.get("distribusi"):
        st.caption("Distribusi B_s: " + ", ".join(f"{k} {v}" for k, v in info["distribusi"].items()))

    a, b = cfg.aturan, cfg.beban
    with st.form("parameter"):
        st.markdown("**Aturan & tafsiran kurikulum**")
        c1, c2, c3 = st.columns(3)
        prasyarat_e = c1.checkbox("Tafsiran literal prasyarat '≥ E' (cukup pernah menempuh)",
                                  value=a.PRASYARAT_E_BERARTI_PERNAH_MENEMPUH,
                                  help="Default nonaktif: AD-03 mensyaratkan MK prasyarat telah LULUS.")
        mode_pilihan = c2.selectbox("Mode penawaran MK pilihan", ["paritas", "tabel11_ketat"],
                                    index=["paritas", "tabel11_ketat"].index(a.MODE_PENAWARAN_PILIHAN))
        ambang = c3.selectbox("Nilai lulus minimum", a.SKALA_NILAI, index=a.SKALA_NILAI.index(a.NILAI_LULUS_MINIMUM))
        penawaran_09 = st.checkbox("MK yang disarankan 'tambah' oleh 09 dianggap ditawarkan pada paritas tersebut",
                                   value=a.PENAWARAN_DARI_REKOMENDASI_09)
        st.markdown("**Beban kognitif & θ**")
        c1, c2, c3, c4 = st.columns(4)
        cara = c1.radio("Penentuan θ", ["otomatis", "manual"], index=0 if b.theta is None else 1)
        theta_manual = c2.number_input("θ manual", 1.0, 10000.0, float(b.theta or ctx.theta or 60.0), 1.0)
        persentil = c3.number_input("Persentil otomatis", 1.0, 100.0, float(b.theta_persentil), 1.0)
        metode = c4.selectbox("Metode beban semester", ["jumlah", "rata_tertimbang"],
                              index=["jumlah", "rata_tertimbang"].index(b.metode))
        c1, c2, c3 = st.columns(3)
        sumber_cl = c1.selectbox("Sumber CL", ["heuristik", "llm", "llm_mock", "pakar"],
                                 index=["heuristik", "llm", "llm_mock", "pakar"].index(b.sumber_cl))
        agregasi = c2.selectbox("Agregasi CL per MK", ["median", "maks", "rata"],
                                index=["median", "maks", "rata"].index(b.agregasi_cl))
        transisi = c3.checkbox("Kecualikan PLP/PKL/PKM/Capstone dari beban", value=b.kecualikan_kegiatan_transisi)
        st.markdown("**Batas SKS per IP semester sebelumnya (KF-11)**: kosongkan kolom = pakai dataset 09")
        pita0 = sorted({(p.ip_min, p.ip_max, p.sks_min, p.sks_max) for p in load_kurikulum(cfg, override=False).pita_sks
                        if p.jalur == "SEMUA"}, reverse=True)
        timpa = a.BATAS_SKS_IP or {}
        batas_df = st.data_editor(pd.DataFrame([{
            "IP min": lo, "IP maks": hi, "SKS min (dataset)": mn, "SKS maks (dataset)": mx,
            "SKS min": (timpa.get(round(lo, 2)) or [None, None])[0], "SKS maks": (timpa.get(round(lo, 2)) or [None, None])[1]}
            for lo, hi, mn, mx in dict.fromkeys(pita0)]).drop_duplicates(["IP min"]), hide_index=True,
            disabled=["IP min", "IP maks", "SKS min (dataset)", "SKS maks (dataset)"], key="ed_batas_sks")
        if st.form_submit_button("Simpan parameter", type="primary"):
            try:
                save_override({
                    "aturan": {"PRASYARAT_E_BERARTI_PERNAH_MENEMPUH": prasyarat_e, "MODE_PENAWARAN_PILIHAN": mode_pilihan,
                               "NILAI_LULUS_MINIMUM": ambang, "PENAWARAN_DARI_REKOMENDASI_09": penawaran_09,
                               "BATAS_SKS_IP": {float(r["IP min"]): [int(r["SKS min"]), int(r["SKS maks"])]
                                                for _, r in batas_df.iterrows()
                                                if pd.notna(r["SKS min"]) and pd.notna(r["SKS maks"])} or None},
                    "beban": {"theta": float(theta_manual) if cara == "manual" else None,
                              "theta_persentil": float(persentil), "metode": metode, "sumber_cl": sumber_cl,
                              "agregasi_cl": agregasi, "kecualikan_kegiatan_transisi": transisi},
                }, ganti=("aturan.BATAS_SKS_IP",))
                simpan_berkas(cfg)
                st.success("Parameter tersimpan.")
                st.rerun()
            except Exception as e:
                st.error(f"Parameter ditolak: {e}")
    if OVERRIDE_PATH.exists() and st.button("Kembalikan ke config.yaml"):
        reset_override()
        simpan_berkas(cfg)
        st.rerun()


def _validasi_krs(cfg) -> None:
    """Validasi rekomendasi semester berjalan terhadap KRS akademik nyata (mahasiswa data uji)."""
    st.markdown("**Validasi terhadap KRS akademik.** Untuk mahasiswa data uji, IP & IPK semester sebelumnya dari data "
                "akademik, MK lulus disimpulkan dari struktur Jalur A (hanya profil yang SKS-nya konsisten). "
                "Rekomendasi semester berjalan dibandingkan dengan KRS nyata.")
    if st.button("Jalankan validasi KRS", disabled=not sumber_data(cfg)["krs_akademik"]):
        from src.simulate import jalankan_validasi_krs

        bar = st.progress(0.0)
        try:
            jalankan_validasi_krs(cfg, progress=lambda i, n: bar.progress(i / n))
        except ValueError as e:
            st.error(str(e))
    r = db.baca_ringkasan(cfg, "validasi_krs")
    if not r or not r.get("n_profil"):
        if r and r.get("dikecualikan"):
            st.caption(f"Tidak ada profil yang memenuhi syarat · dikecualikan: {r['dikecualikan']}")
        return
    rata = r["rata"]
    c = st.columns(4)
    c[0].metric("Profil", r["n_profil"], help=f"dari {r['n_kandidat']} kandidat · {r['per_semester']}")
    c[1].metric("Jaccard (hybrid)", f"{rata['jaccard']['hybrid']:.3f}",
                delta=f"{rata['jaccard']['hybrid'] - rata['jaccard']['arm']:+.3f} vs ARM")
    c[2].metric("Presisi (hybrid)", f"{rata['presisi']['hybrid']:.3f}",
                delta=f"{rata['presisi']['hybrid'] - rata['presisi']['arm']:+.3f} vs ARM")
    c[3].metric("KRS nyata > θ", f"{r['proporsi_aktual_melebihi_theta']:.0%}",
                help=f"Proporsi KRS nyata yang beban kognitifnya melebihi θ = {r['theta']:.1f}")
    st.dataframe(db.baca_hasil(cfg, "validasi_krs_ringkasan"), hide_index=True, width="stretch")
    if r.get("dikecualikan"):
        st.caption("Dikecualikan: " + ", ".join(f"{k} ({n})" for k, n in r["dikecualikan"].items()))
    df = db.baca_hasil(cfg, "validasi_krs")
    if df is not None:
        st.download_button("Unduh validasi_krs.csv (berpasangan untuk JASP)", df.to_csv(index=False).encode("utf-8"),
                           file_name="validasi_krs.csv", mime="text/csv")


def tab_simulasi(cfg) -> None:
    _validasi_krs(cfg)
    st.divider()
    st.markdown("**Simulasi riwayat (UC-12)**: profil dari riwayat multisemester **data uji** dipotong di semester "
                "k; kedua mode dibandingkan pada profil yang sama.")
    _, alasan = transaksi_segar(cfg, "uji")
    ada_uji = alasan is None
    if not ada_uji:
        st.info(f"Butuh data KRS alumni + preprocessing ({alasan}).")
    k_list = st.multiselect("Semester potong k", list(range(1, 8)), default=cfg.simulasi.k_potong)
    if st.button("Jalankan simulasi komparasi", type="primary", disabled=not (k_list and ada_uji)):
        from src.simulate import jalankan_simulasi

        bar = st.progress(0.0)
        jalankan_simulasi(cfg, sorted(k_list), progress=lambda i, n: bar.progress(i / n))
        st.success("Simulasi selesai.")
    r = db.baca_ringkasan(cfg, "simulasi")
    metrik = db.baca_hasil(cfg, "simulasi_metrik")
    if not r or metrik is None or not r.get("n_profil"):
        return
    metrik = metrik.drop(columns="sumber_data", errors="ignore")
    rata = r["rata"]
    c = st.columns(4)
    c[0].metric("Profil", r["n_profil"], help=f"{r['n_alumni_uji']} alumni uji × k {r['k_potong']}")
    c[1].metric("Beban puncak (hybrid)", f"{rata['beban_puncak']['hybrid']:.1f}",
                delta=f"{rata['beban_puncak']['hybrid'] - rata['beban_puncak']['arm']:+.1f} vs ARM", delta_color="inverse")
    c[2].metric("SD beban (hybrid)", f"{rata['sd_beban']['hybrid']:.2f}",
                delta=f"{rata['sd_beban']['hybrid'] - rata['sd_beban']['arm']:+.2f} vs ARM", delta_color="inverse")
    c[3].metric("Jaccard vs alumni (hybrid)", f"{rata['jaccard_rata']['hybrid']:.3f}",
                delta=f"{rata['jaccard_rata']['hybrid'] - rata['jaccard_rata']['arm']:+.3f} vs ARM")
    st.markdown(f"**Rata-rata metrik per mode** (θ = {r['theta']:.1f})")
    st.dataframe(db.baca_hasil(cfg, "simulasi_ringkasan"), hide_index=True, width="stretch")
    gagal = r.get("profil_gagal_aturan_keras", {})
    n_gagal = {m: len(v) for m, v in gagal.items()}
    if any(n_gagal.values()):
        with st.expander(f"Profil yang melanggar AD-02/03/04: {n_gagal}"):
            for m, daftar in gagal.items():
                for x in daftar:
                    st.markdown(f"- **{m}** {x['id_anon']} k={x['k']}: {x[f'pelanggaran_{m}']}")
    else:
        st.success("Kedua mode lolos AD-02, AD-03, dan AD-04 pada semua profil simulasi.")
    with st.expander("Data berpasangan (satu baris per profil)"):
        st.dataframe(metrik, hide_index=True, width="stretch")
    st.download_button("Unduh simulasi_metrik.csv (format berpasangan untuk JASP)",
                       metrik.to_csv(index=False).encode("utf-8"),
                       file_name="simulasi_metrik.csv", mime="text/csv", type="primary")


def _tanpa_asal_data(x):
    """Buang kunci asal data (KUNCI_ASAL_DATA) secara rekursif sebelum log ditampilkan."""
    if isinstance(x, dict):
        return {k: _tanpa_asal_data(v) for k, v in x.items() if k not in KUNCI_ASAL_DATA}
    if isinstance(x, list):
        return [_tanpa_asal_data(v) for v in x]
    return x


def tab_log(cfg) -> None:
    st.markdown("Catatan setiap proses beserta parameter dan waktunya (tabel `log_proses`, KNF-09).")
    log = baca_log(cfg, n=200)
    if not log:
        st.info("Belum ada log.")
        return
    df = pd.DataFrame([{"Mulai": x["mulai"], "Proses": x["proses"], "Status": x["status"],
                        "Durasi (s)": x["durasi_detik"], "run_id": x["run_id"],
                        "Parameter": json.dumps(_tanpa_asal_data(x.get("parameter")), ensure_ascii=False),
                        "Ringkasan": json.dumps(_tanpa_asal_data(x.get("ringkasan")), ensure_ascii=False,
                                                default=str)}
                       for x in reversed(log)])
    st.dataframe(df, hide_index=True, width="stretch")


def halaman_admin(cfg, ctx: KonteksPlanner) -> None:
    st.header("Administrator")
    status = _status_pipeline(versi_data())
    if status["basi"]:
        st.warning("Data masukan berubah sejak pipeline terakhir dijalankan, jalankan ulang preprocessing & ARM.")
    isi = {"Mulai di sini": lambda: tab_mulai(cfg, ctx), "Kurikulum": lambda: tab_kurikulum(cfg, ctx),
           "Data akademik": lambda: tab_data_akademik(cfg), "Preprocessing": lambda: tab_preprocessing(cfg),
           "ARM": lambda: tab_arm(cfg, ctx), "Beban kognitif (LLM)": lambda: tab_beban(cfg, ctx),
           "Parameter": lambda: tab_parameter(cfg, ctx), "Simulasi & validasi": lambda: tab_simulasi(cfg),
           "Log": lambda: tab_log(cfg)}
    # on_change="rerun": hanya isi tab yang sedang dibuka yang dijalankan (bukan kesembilan tab setiap klik)
    for tab, tampil in zip(st.tabs(list(isi), key="tab_admin", on_change="rerun"), isi.values()):
        if tab.open:
            with tab:
                tampil()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _pulihkan_sekali(lokasi: str) -> list[str]:
    """Hosting tanpa disk persisten: pulihkan riwayat & parameter UI dari basis data sekali per proses."""
    return pulihkan_berkas(load_config())


def main() -> None:
    cfg = load_config()
    if _pulihkan_sekali(f"{db.keterangan(cfg)}|{cfg.raw_dir}"):
        cfg = load_config()      # config.override.yaml mungkin ikut dipulihkan
    st.title("Time Allocation Optimizer")
    st.caption("Sistem Optimasi Alokasi Waktu & Rencana Studi · Association Rule Mining (FP-Growth) + Beban Kognitif Bloom (LLM)")
    st.session_state["_versi_data"] = _versi(cfg)
    banner(cfg)

    with st.sidebar:
        peran = st.radio("Peran", ["Mahasiswa", "DPA", "Administrator"])
        if auth.perlu_masuk(peran, cfg) and st.session_state.get(f"masuk_{peran}"):
            if st.button("Keluar"):
                st.session_state.pop(f"masuk_{peran}", None)
                st.rerun()

    if not gerbang_peran(peran, cfg):
        return
    ctx = _konteks(versi_data())

    with st.sidebar:
        st.divider()
        st.markdown("**Status sistem**")
        status = _status_data(versi_data())
        sd, ak = status["sumber"], status["akademik"]
        st.caption("KRS akademik: " + (f"{', '.join(ak.get('periode_krs', []))} ({ak.get('n_mahasiswa_krs', '–')} "
                                       f"mahasiswa)" if sd["krs_akademik"] else "belum diimpor"))
        st.caption(f"Riwayat multisemester: {'tersedia' if sd['riwayat'] else 'belum tersedia'}")
        n_aturan = sum(len(v) for v in ctx.arm.per_konsekuen.values())
        st.caption(f"ARM: {n_aturan} aturan (konsekuen tunggal)" if not ctx.arm.kosong
                   else f"ARM: belum aktif ({ctx.arm.catatan or 'tanpa data'})")
        st.caption(f"CL: {ctx.info_cl.get('sumber_rating', '–')} · agregasi {cfg.beban.agregasi_cl}")
        st.caption(f"θ = {ctx.theta:.1f} ({'dikalibrasi data KRS' if ctx.info_theta.get('dikalibrasi_data') else 'belum dikalibrasi data KRS'})"
                   if ctx.theta is not None else "θ belum tersedia")
        st.caption(f"LLM: {cfg.llm.provider} · {cfg.llm.model}")
        st.caption(f"Basis data: {db.keterangan(cfg)}")

    if peran == "Mahasiswa":
        halaman_mahasiswa(cfg, ctx)
    elif peran == "DPA":
        halaman_dpa(cfg, ctx)
    else:
        halaman_admin(cfg, ctx)


main()
