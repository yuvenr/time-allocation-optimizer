"""Komponen tampilan bersama untuk UI Streamlit (tanpa CSS kustom)."""
from __future__ import annotations

from typing import Callable

import altair as alt
import pandas as pd
import streamlit as st

from src.metrics import metrik_rencana
from src.penyesuaian import (kembalikan_rekomendasi, pindahkan, semester_papan, semester_tujuan_sah,
                             sesuaikan_rencana, susunan_rencana)
from src.rules import DEFINISI_ATURAN
from src.schemas import Plan

try:   # komponen drag & drop (iterasi 3); tanpa paket ini papan diganti menu pindah
    from streamlit_sortables import sort_items
except ImportError:  # pragma: no cover
    sort_items = None

LABEL_MODE = {"arm": "ARM murni", "hybrid": "Hybrid (ARM + beban kognitif)"}
LABEL_SINGKAT = {"arm": "ARM murni", "hybrid": "Hybrid"}
LABEL_JENIS = {"paket": "paket", "reguler": "reguler", "konversi": "konversi merdeka belajar"}

# Slot kategorikal 1 & 2 dari palet referensi (tervalidasi CVD di mode terang & gelap).
WARNA = {
    "light": {"arm": "#2a78d6", "hybrid": "#eb6834", "tinta": "#52514e"},
    "dark": {"arm": "#3987e5", "hybrid": "#d95926", "tinta": "#c3c2b7"},
}


def _tema() -> dict:
    try:
        jenis = st.context.theme.type or "light"
    except Exception:
        jenis = "light"
    return WARNA["dark" if jenis == "dark" else "light"]


def grafik_beban(rencana: list[Plan], theta: float | None) -> None:
    """Bar beban kognitif per semester + garis θ. Satu rencana → satu seri tanpa legenda;
    dua rencana → bar berkelompok dengan legenda."""
    warna = _tema()
    baris = []
    for p in rencana:
        for s in p.semester:
            baris.append({"semester": f"S{s.semester}", "urut": s.semester, "beban": round(s.total_beban, 1),
                          "sks": s.total_sks, "mk": len(s.items), "jenis": LABEL_JENIS.get(s.jenis, s.jenis),
                          "mode": LABEL_SINGKAT[p.mode]})
    df = pd.DataFrame(baris)
    if df.empty:
        return
    urutan = [f"S{i}" for i in sorted(df["urut"].unique())]
    tooltip = [alt.Tooltip("mode:N", title="Mode"), alt.Tooltip("semester:N", title="Semester"),
               alt.Tooltip("jenis:N", title="Jenis"), alt.Tooltip("beban:Q", title="Beban", format=".1f"),
               alt.Tooltip("sks:Q", title="SKS"), alt.Tooltip("mk:Q", title="Jumlah MK")]
    x = alt.X("semester:N", sort=urutan, title="Semester", axis=alt.Axis(labelAngle=0))
    puncak = max(float(df["beban"].max()), theta or 0.0, 1.0)
    skala_y = alt.Scale(domain=[0, puncak * 1.15], nice=True)  # ruang untuk label θ
    y = alt.Y("beban:Q", title="Beban kognitif semester (Σ CL × SKS)", scale=skala_y)
    batang = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    if len(rencana) == 1:
        batang = batang.encode(x=x, y=y, color=alt.value(warna[rencana[0].mode]), tooltip=tooltip)
    else:
        mode_ada = [LABEL_SINGKAT[p.mode] for p in rencana]
        batang = batang.encode(
            x=x, y=y, xOffset=alt.XOffset("mode:N", sort=mode_ada, scale=alt.Scale(paddingInner=0.08)),
            color=alt.Color("mode:N", sort=mode_ada, title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=mode_ada, range=[warna[p.mode] for p in rencana])),
            tooltip=tooltip,
        )
    lapisan = [batang]
    if theta is not None:
        garis = alt.Chart(pd.DataFrame({"theta": [theta]}))
        lapisan.append(garis.mark_rule(strokeDash=[6, 4], strokeWidth=2, color=warna["tinta"]).encode(
            y=alt.Y("theta:Q", scale=skala_y), tooltip=[alt.Tooltip("theta:Q", title="θ", format=".1f")]))
        lapisan.append(garis.mark_text(align="left", baseline="bottom", dx=4, dy=-4, color=warna["tinta"]).encode(
            y=alt.Y("theta:Q", scale=skala_y), x=alt.value(0), text=alt.value(f"θ = {theta:.1f}")))
    st.altair_chart(alt.layer(*lapisan).properties(height=320), width="stretch")


def tabel_validasi(plan: Plan) -> None:
    v = plan.validasi
    if v is None:
        return
    verifikasi = {h.kode for h in v.rincian if h.status == "perlu_verifikasi_manual"}

    def status(kode: str) -> str:
        if not v.per_aturan.get(kode, True):
            return "❌ dilanggar"
        return "⚠️ perlu verifikasi manual" if kode in verifikasi else "✅ lolos"

    df = pd.DataFrame([{"Aturan": k, "Keterangan": nama, "Rujukan": rujukan, "Status": status(k)}
                       for k, (nama, rujukan) in DEFINISI_ATURAN.items()])
    st.dataframe(df, hide_index=True, width="stretch")
    if v.pelanggaran:
        with st.expander(f"Rincian pelanggaran ({len(v.pelanggaran)})", expanded=True):
            for p in v.pelanggaran:
                st.markdown(f"- {p}")
    if v.perlu_verifikasi:
        with st.expander(f"Perlu verifikasi manual oleh DPA ({len(v.perlu_verifikasi)})"):
            st.caption("Syarat yang isinya tidak termuat di dokumen kurikulum, tidak dianggap otomatis terpenuhi.")
            for p in v.perlu_verifikasi:
                st.markdown(f"- {p}")


def _tabel_semester(s, kurikulum) -> pd.DataFrame:
    return pd.DataFrame([{
        "Kode": i.kode if i.jenis == "mk" else "-", "Mata kuliah": i.nama, "SKS": i.sks, "Sifat": i.sifat,
        "Tema": i.tema or (", ".join(i.tema_konversi) if i.tema_konversi else ""),
        "CL": round(i.cl, 2) if i.cl is not None else None, "Prioritas": i.prioritas or "",
    } for i in s.items])


def _label_papan(kode: str, kur) -> str:
    return f"{kode} · {kur.mk[kode].nama} ({kur.sks(kode)} SKS)"


def _terapkan(hasil, kunci: str, simpan: Callable[[Plan], None]) -> None:
    """Simpan hasil penyesuaian (bila diterima), siapkan pesan, reset papan, lalu muat ulang."""
    if hasil.diterima and hasil.berubah:
        simpan(hasil.plan)
        pindah = ", ".join(f"{p.kode} S{p.dari}→S{p.ke}" for p in hasil.pemindahan)
        st.session_state[f"pesan-papan-{kunci}"] = ("sukses", f"Pemindahan diterima: {pindah}.", hasil.catatan)
    elif not hasil.diterima:
        st.session_state[f"pesan-papan-{kunci}"] = (
            "tolak", "Pemindahan ditolak karena melanggar aturan kurikulum, susunan dikembalikan:", hasil.ditolak)
    st.session_state[f"versi-papan-{kunci}"] = st.session_state.get(f"versi-papan-{kunci}", 0) + 1
    st.rerun()


def papan_penyesuaian(plan: Plan, ctx, kunci: str, simpan: Callable[[Plan], None]) -> None:
    """Iterasi 3: mahasiswa menggeser MK antarsemester (drag & drop, atau menu pindah yang aksesibel).
    Setiap susunan baru diperiksa ``src.penyesuaian`` dengan validator yang sama dengan planner."""
    kur = ctx.kurikulum
    sus = susunan_rencana(plan)
    if not sus:
        return
    st.markdown("**Sesuaikan rencana: geser MK antarsemester**")
    st.caption("Seret MK ke semester lain (mis. menunda PLP, PKM, atau Tugas Akhir). Setiap perubahan langsung diperiksa "
               "terhadap prasyarat, semester penawaran, batas SKS, syarat Tugas Akhir/PLP, dan masa studi; pemindahan "
               "yang melanggar ditolak beserta alasannya. Semester paket dan konversi tidak dapat diubah. PLP 1 dan "
               "PLP 2 selalu dipindahkan bersama.")
    if pesan := st.session_state.pop(f"pesan-papan-{kunci}", None):
        jenis, judul, rincian = pesan
        (st.success if jenis == "sukses" else st.error)(judul + "".join(f"\n- {x}" for x in rincian))
    versi = st.session_state.get(f"versi-papan-{kunci}", 0)
    semester = semester_papan(plan, ctx)
    total = {sm.semester: sm for sm in plan.semester}

    def judul(s: int) -> str:
        sm = total.get(s)
        if sm is None:
            return f"S{s} (semester tambahan, kosong)"
        return f"S{s}: {sm.total_sks}/{sm.batas_sks} SKS · beban {sm.total_beban:.1f}"

    if sort_items is not None:
        label = {k: _label_papan(k, kur) for ks in sus.values() for k in ks}
        balik = {v: k for k, v in label.items()}
        wadah = [{"header": judul(s), "items": [label[k] for k in sus.get(s, [])]} for s in semester]
        hasil = sort_items(wadah, multi_containers=True, direction="vertical", key=f"dnd-{kunci}-{plan.id}-{versi}")
        baru = {s: [balik[x] for x in w["items"]] for s, w in zip(semester, hasil or wadah)}
        if any(set(baru.get(s, [])) != set(sus.get(s, [])) for s in semester):
            _terapkan(sesuaikan_rencana(plan, baru, ctx), kunci, simpan)
    else:
        st.caption("Komponen drag & drop (`streamlit-sortables`) belum terpasang, gunakan menu di bawah.")

    with st.expander("Pindahkan lewat menu", expanded=sort_items is None):
        dapat = [k for s in sorted(sus) for k in sus[s]]
        asal = {k: s for s, ks in sus.items() for k in ks}
        c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
        kode = c1.selectbox("Mata kuliah", dapat, key=f"menu-mk-{kunci}-{versi}",
                            format_func=lambda k: f"S{asal[k]} · {_label_papan(k, kur)}")
        sah = semester_tujuan_sah(plan, kode, ctx)
        ke = c2.selectbox("Ke semester", sah, key=f"menu-ke-{kunci}-{versi}", format_func=lambda s: f"S{s}",
                          disabled=not sah)
        if c3.button("Pindahkan", key=f"menu-pindah-{kunci}", disabled=not sah, width="stretch"):
            _terapkan(pindahkan(plan, kode, ke, ctx), kunci, simpan)
        st.caption("Hanya semester yang memenuhi seluruh aturan keras yang ditampilkan." if sah else
                   "Tidak ada semester lain yang memenuhi aturan untuk MK ini (penawaran, prasyarat, atau batas SKS).")
    if plan.disesuaikan and st.button("Kembalikan ke rekomendasi sistem", key=f"reset-papan-{kunci}"):
        simpan(kembalikan_rekomendasi(plan, ctx))
        st.session_state[f"versi-papan-{kunci}"] = versi + 1
        st.rerun()


def ringkasan_penyesuaian(plan: Plan, kurikulum) -> None:
    """Untuk mahasiswa & DPA: daftar pemindahan manual + susunan asli rekomendasi sistem."""
    if not plan.disesuaikan:
        return
    with st.expander(f"Disesuaikan mahasiswa: {len(plan.penyesuaian)} pemindahan (bandingkan dengan rekomendasi "
                     f"sistem)"):
        if plan.penyesuaian:
            st.dataframe(pd.DataFrame([{
                "Waktu": p.waktu.strftime("%d-%m-%Y %H:%M"), "Kode": p.kode,
                "Mata kuliah": kurikulum.mk[p.kode].nama if p.kode in kurikulum else p.kode,
                "Dari": f"S{p.dari}", "Ke": f"S{p.ke}", "Keterangan": "korequisit (otomatis)" if p.otomatis else "",
            } for p in plan.penyesuaian]), hide_index=True, width="stretch")
        sistem = {sm.semester: [i.kode for i in sm.items] for sm in plan.semester_sistem or []}
        kini = {sm.semester: [i.kode for i in sm.items] for sm in plan.semester}
        st.dataframe(pd.DataFrame([{
            "Semester": f"S{s}", "Rekomendasi sistem": ", ".join(sistem.get(s, [])) or "–",
            "Setelah disesuaikan": ", ".join(kini.get(s, [])) or "–",
            "Berubah": "ya" if set(sistem.get(s, [])) != set(kini.get(s, [])) else "",
        } for s in sorted(set(sistem) | set(kini))]), hide_index=True, width="stretch")


def tampilkan_rencana(plan: Plan, kurikulum, cfg, kunci: str, ctx=None,
                      simpan: Callable[[Plan], None] | None = None) -> None:
    """Detail satu rencana: ringkasan, asumsi, grafik beban, (papan penyesuaian bila ``simpan`` diberikan),
    tabel per semester, validasi, alasan."""
    m = metrik_rencana(plan, cfg, theta=plan.theta)
    sks_awal = kurikulum.total_sks(plan.profil.mk_lulus) + plan.profil.sks_konversi
    min_lulus = kurikulum.param_int("AT01", "total_min")
    k = st.columns(5)
    k[0].metric("SKS direncanakan", m["total_sks_rencana"])
    k[1].metric("Total SKS akhir", sks_awal + m["total_sks_rencana"],
                help=f"SKS lulus saat ini {sks_awal} + rencana. Minimal lulus {min_lulus} SKS (AT01).")
    k[2].metric("Beban puncak", f"{m['beban_puncak']:.1f}")
    k[3].metric("θ", f"{plan.theta:.1f}" if plan.theta is not None else "–",
                help="Ambang beban per semester. Hanya membatasi mode hybrid.")
    k[4].metric("Validasi aturan", "Lolos" if plan.validasi and plan.validasi.lolos else "Ada pelanggaran")

    for w in plan.peringatan:
        st.warning(w)
    with st.expander(f"Asumsi rencana ({len(plan.asumsi)})"):
        for a in plan.asumsi:
            st.markdown(f"- {a}")

    st.markdown("**Beban kognitif per semester**")
    grafik_beban([plan], plan.theta)
    ringkasan_penyesuaian(plan, kurikulum)
    if ctx is not None and simpan is not None:
        papan_penyesuaian(plan, ctx, kunci, simpan)

    st.markdown("**Rencana per semester**")
    for s in plan.semester:
        judul = f"Semester {s.semester} ({LABEL_JENIS.get(s.jenis, s.jenis)}), {s.total_sks} SKS"
        if s.batas_sks is not None:
            judul += f" / batas {s.batas_sks}"
        judul += f" · beban {s.total_beban:.1f}"
        st.markdown(f"*{judul}*")
        if not s.items:
            st.caption("Tidak ada MK yang direncanakan (tidak ada MK yang dibutuhkan dan ditawarkan).")
            continue
        st.dataframe(_tabel_semester(s, kurikulum), hide_index=True, width="stretch",
                     column_config={"CL": st.column_config.NumberColumn(
                         help="Beban kognitif per MK: agregasi level Bloom CPMK kognitif (1–6)")})
        for c in s.catatan:
            st.caption(c)
        if s.dilewati_theta:
            st.caption("Dilewati karena beban > θ: " + ", ".join(kurikulum.label(x) for x in s.dilewati_theta))
        with st.expander(f"Alasan, sumber & jadwal per MK semester {s.semester}"):
            if s.sumber_batas:
                st.caption(f"Batas SKS: {s.sumber_batas}")
            for i in s.items:
                st.markdown(f"**{i.kode if i.jenis == 'mk' else 'Konversi'} - {i.nama}**")
                for a in i.alasan:
                    st.markdown(f"- {a}")
                if i.justifikasi_cpmk:
                    st.markdown("- Level Bloom per CPMK (justifikasi penilaian):")
                    for j in i.justifikasi_cpmk:
                        st.markdown(f"    - {j}")
                for j in i.jadwal:
                    st.markdown(f"- 🗓️ {j}")
                for p in i.perlu_verifikasi:
                    st.markdown(f"- ⚠️ {p}")
                if i.sumber:
                    st.caption("Sumber aturan: " + " · ".join(i.sumber))

    st.markdown("**Status validasi aturan domain**")
    tabel_validasi(plan)
    st.download_button("Unduh rencana (JSON)", plan.model_dump_json(indent=2, exclude={"sumber_data"}), file_name=f"rencana-{plan.id}.json",
                       mime="application/json", key=f"unduh-{kunci}")


def bandingkan(rencana: dict[str, Plan], cfg) -> None:
    """Perbandingan mode arm vs hybrid untuk profil yang sama."""
    a, h = rencana["arm"], rencana["hybrid"]
    ma, mh = metrik_rencana(a, cfg, theta=h.theta), metrik_rencana(h, cfg, theta=h.theta)
    k = st.columns(4)
    k[0].metric("Beban puncak (hybrid)", f"{mh['beban_puncak']:.1f}",
                delta=f"{mh['beban_puncak'] - ma['beban_puncak']:+.1f} vs ARM", delta_color="inverse")
    k[1].metric("SD beban (hybrid)", f"{mh['sd_beban']:.1f}",
                delta=f"{mh['sd_beban'] - ma['sd_beban']:+.1f} vs ARM", delta_color="inverse")
    k[2].metric("SD SKS (hybrid)", f"{mh['sd_sks']:.2f}",
                delta=f"{mh['sd_sks'] - ma['sd_sks']:+.2f} vs ARM", delta_color="inverse")
    k[3].metric("Semester terisi (hybrid)", mh["n_semester_terisi"],
                delta=f"{mh['n_semester_terisi'] - ma['n_semester_terisi']:+d} vs ARM", delta_color="off")
    grafik_beban([a, h], h.theta)
    df = pd.DataFrame([
        {"Metrik": nama, "ARM murni": ma[kunci], "Hybrid": mh[kunci]}
        for nama, kunci in [("SD beban", "sd_beban"), ("Beban puncak", "beban_puncak"), ("Rata-rata beban", "rata_beban"),
                            ("SD SKS", "sd_sks"), ("Total SKS rencana", "total_sks_rencana"),
                            ("Semester terisi", "n_semester_terisi"), ("Lolos semua aturan", "lolos_semua"),
                            ("Semua semester ≤ θ", "lolos_theta")]
    ])
    df[["ARM murni", "Hybrid"]] = df[["ARM murni", "Hybrid"]].astype(str)
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("Kedua mode memakai algoritma yang sama; perbedaan hanya berasal dari batasan beban θ pada mode hybrid.")
