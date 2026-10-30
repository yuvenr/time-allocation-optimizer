"""Beban kognitif: penilaian Bloom per CPMK → CL per MK (UC-09; CLAUDE.md §7).

Sumber penilaian (label identik dengan instrumen pakar DPA: C1–C6, ``A`` = afektif):
- ``heuristik``: ``bloom_maks_heuristik`` di ``02_cpmk.csv`` (kamus KKO 18) — baseline, BUKAN ground truth;
- ``llm`` / ``llm_mock``: LLM-as-judge;
- ``pakar``: penilaian DPA (diisi lewat formulir terpisah, lalu diimpor — UC-10; ground truth).
Rating disimpan di tabel ``rating_bloom`` (SQLite), terpisah dari dataset, dan di-*join* lewat
``id_cpmk`` sehingga regenerasi dataset tidak menimpanya (CLAUDE.md §2). Kolom ``penilai`` berisi
nomor pengulangan (LLM) atau kode pakar.

Aturan permintaan LLM (prompt v3, BATCH):
- Satu permintaan berisi SELURUH MK (``llm.mk_per_batch = 0``; atau potongan berukuran tetap) per pengulangan,
  ``n_pengulangan`` (3) kali → 3 permintaan untuk 70 MK, bukan 210. Instruksi sistem & few-shot cukup dikirim
  sekali per batch. Keluaran = satu objek JSON schema ketat berisi penilaian per CPMK berlabel lokal
  (``MK07-CPMK03``) yang kemudian dipisah kembali per MK (``pisah_per_mk``).
- Label lokal menyembunyikan kode MK (yang membocorkan semester); prompt tidak memuat SKS, semester, skor
  heuristik/pakar, atau skor lain. Urutan MK dan urutan tiga contoh few-shot fiktif (rendah/sedang/tinggi,
  dengan contoh penalaran) diacak per pengulangan dengan seed tetap — mengurangi bias posisi antar-batch.
- MK yang tidak lengkap/tidak sah dikirim ulang sendiri (≤ ``llm.max_retry``) lalu ``perlu_tinjau_manual``;
  keluaran terpotong (batas token) → batch dibelah dua. Setiap permintaan tercatat di tabel ``batch_llm``
  (token, status, percobaan) dan setiap CPMK di ``rating_bloom`` (penilai = nomor pengulangan), sehingga
  konsistensi antarpengulangan dianalisis per CPMK (``konsistensi_pengulangan``). Dapat dilanjutkan.
- Hasil prompt v2 (satu permintaan per MK) yang sudah tersimpan tetap dipakai apa adanya.

CL per MK = agregasi (``beban.agregasi_cl``: median | maks | rata) level Bloom CPMK kognitif
(C1..C6 → 1..6) setelah konsensus antar-pengulangan/pakar; CPMK ``A`` tidak ikut dihitung. Cara
agregasi ini adalah ASUMSI dan dicatat di setiap rencana.

Jalankan mandiri:  python -m src.llm.beban [--baru]
"""
from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src import db
from src.config import Config, load_config
from src.data_loader import Kurikulum, load_kurikulum
from src.llm.client import (LLMClient, LLMError, LLMKonfigurasiError, LLMResponse, MockClient, buat_client,
                            parse_json_keluaran)
from src.llm.konteks import KonteksMK, SumberKonteks
from src.llm.prompts import TemplatPrompt, muat_prompt
from src.logutil import catat_proses, teruskan_log_sdk
from src.schemas import LEVEL_BLOOM, PenilaianBloom

SUMBER_TERSIMPAN = ("llm", "llm_mock", "pakar")
KOLOM_RATING = ["sumber", "id_cpmk", "kode_mk", "penilai", "model", "versi_prompt", "bloom", "ranah", "penalaran",
                "alasan", "keyakinan", "provider", "model_pelayan", "temperature", "thinking", "ringkasan_thinking",
                "token_masuk", "token_keluar", "status", "percobaan", "catatan", "timestamp"]
KOLOM_PAKAR = ["id_cpmk", "kode_mk", "kode_pakar", "bloom", "catatan"]   # formulir isian pakar (KD-07)

# Few-shot: tiga MATA KULIAH FIKTIF (bukan dari kurikulum) dengan skor acuan
# berdasarkan rubrik — tingkat rendah, sedang, tinggi — masing-masing dengan contoh penalaran (chain-of-thought).
# Urutan contoh diacak per (MK, pengulangan) untuk menghindari bias urutan & label mayoritas (Zhao et al., 2021).
# Contoh "sedang" memuat satu CPMK afektif agar format label "A" terkalibrasi.
CONTOH_FEWSHOT = [
    {"tingkat": "rendah", "nama": "Pengenalan Perangkat Laboratorium Sekolah",
     "deskripsi": "Mata kuliah fiktif tentang jenis dan fungsi perangkat keras di laboratorium komputer sekolah.",
     "cpmk": [
         ("Mahasiswa mampu menyebutkan jenis-jenis perangkat keras pada laboratorium sekolah.",
          {"penalaran": "Kata kerja 'menyebutkan' + objek jenis perangkat keras; prosesnya mengingat kembali fakta "
                        "tanpa pengolahan.", "bloom": "C1", "ranah": "kognitif", "keyakinan": 0.9,
           "alasan": "'Menyebutkan' hanya menuntut mengingat kembali fakta."}),
         ("Mahasiswa mampu menjelaskan fungsi setiap perangkat jaringan di laboratorium.",
          {"penalaran": "Kata kerja 'menjelaskan' + objek fungsi perangkat; menuntut membangun makna dengan kata-kata "
                        "sendiri, belum menerapkan.", "bloom": "C2", "ranah": "kognitif", "keyakinan": 0.85,
           "alasan": "'Menjelaskan' fungsi berarti memahami konsep."}),
     ]},
    {"tingkat": "sedang", "nama": "Praktik Pengolahan Data Nilai",
     "deskripsi": "Mata kuliah fiktif tentang penerapan algoritma dan analisis sederhana pada data nilai siswa.",
     "cpmk": [
         ("Mahasiswa mampu menerapkan algoritma pengurutan pada data nilai ujian siswa.",
          {"penalaran": "Kata kerja 'menerapkan' + objek algoritma pada data nyata; menjalankan prosedur yang sudah "
                        "dikenal pada situasi tertentu.", "bloom": "C3", "ranah": "kognitif", "keyakinan": 0.85,
           "alasan": "'Menerapkan' prosedur pada data tertentu adalah level menerapkan."}),
         ("Mahasiswa mampu membandingkan pola nilai antarkelas untuk menemukan penyebab perbedaannya.",
          {"penalaran": "Kata kerja 'membandingkan' dan 'menemukan penyebab'; menguraikan data dan mencari hubungan "
                        "antarbagian.", "bloom": "C4", "ranah": "kognitif", "keyakinan": 0.8,
           "alasan": "Membandingkan pola untuk menemukan hubungan sebab adalah menganalisis."}),
         ("Mahasiswa mampu menunjukkan sikap jujur dan bertanggung jawab dalam mengolah data nilai.",
          {"penalaran": "Kata kerja 'menunjukkan sikap' + objek nilai kejujuran; menuntut sikap, bukan proses "
                        "kognitif.", "bloom": "A", "ranah": "afektif", "keyakinan": 0.85,
           "alasan": "Menuntut sikap dan nilai, bukan proses kognitif."}),
     ]},
    {"tingkat": "tinggi", "nama": "Proyek Media Pembelajaran Interaktif",
     "deskripsi": "Mata kuliah fiktif berbasis proyek untuk menilai dan mengembangkan media pembelajaran.",
     "cpmk": [
         ("Mahasiswa mampu menilai kelayakan media pembelajaran berdasarkan kriteria kualitas yang ditetapkan.",
          {"penalaran": "Kata kerja 'menilai kelayakan' + 'berdasarkan kriteria'; membuat keputusan berdasarkan "
                        "standar.", "bloom": "C5", "ranah": "kognitif", "keyakinan": 0.85,
           "alasan": "Menilai berdasarkan kriteria adalah mengevaluasi."}),
         ("Mahasiswa mampu merancang aplikasi kuis interaktif untuk pembelajaran sejarah.",
          {"penalaran": "Kata kerja 'merancang' + objek aplikasi baru; menghasilkan produk orisinal.",
           "bloom": "C6", "ranah": "kognitif", "keyakinan": 0.9,
           "alasan": "'Merancang' produk baru menuntut level mencipta."}),
     ]},
]


def rng_batch(seed: int, run: int, salt: str = "") -> np.random.Generator:
    """RNG deterministik per pengulangan (urutan few-shot & urutan MK di dalam batch)."""
    return np.random.default_rng([seed, run, zlib.crc32(salt.encode("utf-8"))])


# ---------------------------------------------------------------------
# Prompt batch, skema, validasi, mock
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PetaBatch:
    """Label lokal di prompt ↔ identitas asli. Label MK (``MK01``) & CPMK (``MK01-CPMK02``) menyembunyikan kode
    MK (yang membocorkan semester) dan memungkinkan hasil dipisah kembali per MK."""

    cpmk: dict[str, tuple[str, str]]       # label CPMK → (kode_mk, id_cpmk)
    mk: dict[str, str]                     # label MK → kode_mk

    def label_mk(self, kode: str) -> list[str]:
        return [lb for lb, (k, _) in self.cpmk.items() if k == kode]


def _blok_contoh(rng: np.random.Generator) -> str:
    contoh = []
    for n, idx in enumerate(rng.permutation(len(CONTOH_FEWSHOT)), start=1):
        c = CONTOH_FEWSHOT[idx]
        label = [f"CONTOH{n}-CPMK{j:02d}" for j in range(1, len(c["cpmk"]) + 1)]
        daftar = "\n".join(f"[{lb}] {teks}" for lb, (teks, _) in zip(label, c["cpmk"]))
        jawaban = {"penilaian": [{"id_cpmk": lb, **j} for lb, (_, j) in zip(label, c["cpmk"])]}
        contoh.append(f"### CONTOH{n}\nNama: {c['nama']}\nDeskripsi: {c['deskripsi']}\nCPMK yang dinilai:\n"
                      f"{daftar}\nJawaban:\n```json\n{json.dumps(jawaban, ensure_ascii=False)}\n```")
    return "\n\n".join(contoh)


def bangun_prompt_batch(mks: list[KonteksMK], rng: np.random.Generator,
                        templat: TemplatPrompt) -> tuple[str, str, PetaBatch]:
    """(system, user, peta). Seluruh MK dalam satu permintaan; urutan MK dan contoh few-shot diacak (rng).
    Tingkat contoh (rendah/sedang/tinggi) tidak ditampilkan agar tidak menjadi jangkar."""
    urut = [mks[i] for i in rng.permutation(len(mks))]
    contoh = _blok_contoh(rng)
    peta_cpmk: dict[str, tuple[str, str]] = {}
    peta_mk: dict[str, str] = {}
    blok = []
    for n, k in enumerate(urut, start=1):
        lmk = f"MK{n:02d}"
        peta_mk[lmk] = k.kode
        baris = []
        for j, (id_cpmk, teks) in enumerate(k.cpmk, start=1):
            lb = f"{lmk}-CPMK{j:02d}"
            peta_cpmk[lb] = (k.kode, id_cpmk)
            baris.append(f"[{lb}] {teks}")
        sub = "\n".join(f"- {teks}" for _, teks in k.subcpmk) or "-"
        blok.append(f"### {lmk}\nNama: {k.nama}\nDeskripsi: {k.deskripsi or '-'}\nSub-CPMK (konteks, tidak dinilai):\n"
                    f"{sub}\nCPMK yang dinilai:\n" + "\n".join(baris))
    user = templat.user.substitute(contoh=contoh, n_mk=len(urut), n_cpmk=len(peta_cpmk),
                                   mata_kuliah="\n\n".join(blok))
    return templat.system, user, PetaBatch(peta_cpmk, peta_mk)


def skema_penilaian(ids: list[str]) -> dict:
    """JSON schema ketat keluaran penilaian (structured outputs): kontrak yang sama dengan ``PenilaianBloom``;
    ``id_cpmk`` dibatasi ke label CPMK di batch. Kelengkapan per MK tetap diperiksa ``pisah_per_mk``."""
    butir = {
        "type": "object",
        "properties": {
            "id_cpmk": {"type": "string", "enum": list(ids)},
            "penalaran": {"type": "string"},
            "bloom": {"type": "string", "enum": [*LEVEL_BLOOM, "A"]},
            "ranah": {"type": "string", "enum": ["kognitif", "afektif", "afektif+kognitif"]},
            "alasan": {"type": "string"},
            "keyakinan": {"type": ["number", "null"]},
        },
        "required": ["id_cpmk", "penalaran", "bloom", "ranah", "alasan", "keyakinan"],
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {"penilaian": {"type": "array", "items": butir}},
            "required": ["penilaian"], "additionalProperties": False}


def pisah_per_mk(obj: dict, peta: PetaBatch) -> tuple[dict[str, list[PenilaianBloom]], list[str]]:
    """Pisahkan keluaran batch per MK. Sebuah MK diterima hanya bila SETIAP CPMK-nya muncul tepat sekali dan
    lolos skema ``PenilaianBloom``; butir tidak sah/asing/ganda membuat MK pemiliknya ditolak (diulang).
    Mengembalikan ({kode_mk: penilaian berlabel lokal}, catatan)."""
    daftar = obj.get("penilaian") if isinstance(obj, dict) else None
    if not isinstance(daftar, list):
        raise ValueError("kunci 'penilaian' harus berupa daftar")
    per_label: dict[str, list] = {}
    catatan: list[str] = []
    for x in daftar:
        lb = x.get("id_cpmk") if isinstance(x, dict) else None
        if lb not in peta.cpmk:
            catatan.append(f"ID asing {lb!r} diabaikan")
            continue
        try:
            per_label.setdefault(lb, []).append(PenilaianBloom.model_validate(x))
        except ValidationError as e:
            per_label.setdefault(lb, []).append(None)
            catatan.append(f"{lb}: {e.errors()[0]['msg']}")
    hasil: dict[str, list[PenilaianBloom]] = {}
    for kode in peta.mk.values():
        label = peta.label_mk(kode)
        nilai = [per_label.get(lb, []) for lb in label]
        if all(len(v) == 1 and v[0] is not None for v in nilai):
            hasil[kode] = [v[0] for v in nilai]
    return hasil, catatan


def _ranah_heuristik(bloom: str, ranah: str) -> str:
    if bloom == "A":
        return "afektif"
    return "afektif+kognitif" if "afektif" in str(ranah) else "kognitif"


def peta_heuristik(sk: SumberKonteks) -> dict[str, tuple[str, str, str]]:
    """id_cpmk → (bloom, ranah, KKO) dari kolom heuristik 02."""
    df = sk.cpmk
    return {r.id_cpmk: (r.bloom_maks_heuristik, _ranah_heuristik(r.bloom_maks_heuristik, r.ranah_heuristik),
                        r.kko_semua if "kko_semua" in df.columns else "")
            for r in df.itertuples()}


def penjawab_mock_bloom(heur: dict[str, tuple[str, str, str]], peta: PetaBatch) -> Callable[[str, str], str]:
    """Mock deterministik: jawaban = heuristik KKO (02) dalam format JSON yang sama dengan LLM."""
    def jawab(system: str, user: str) -> str:
        penilaian = []
        for lokal, (_, id_cpmk) in peta.cpmk.items():
            bloom, ranah, kko = heur[id_cpmk]
            penilaian.append({"id_cpmk": lokal, "penalaran": f"Kata kerja operasional: {kko or '-'} → kamus KKO.",
                              "bloom": bloom, "ranah": ranah, "keyakinan": None,
                              "alasan": f"Heuristik kamus KKO (18_kko_bloom_map): {kko or '-'}."})
        return "Penilaian mock (heuristik).\n```json\n" + json.dumps({"penilaian": penilaian}, ensure_ascii=False) \
            + "\n```"
    return jawab


@dataclass
class HasilBatch:
    """Keluaran satu batch (satu pengulangan): penilaian per MK yang lengkap + jejak permintaan."""

    penilaian: dict[str, list[PenilaianBloom]] = field(default_factory=dict)
    gagal: list[str] = field(default_factory=list)          # kode MK yang tetap tidak lengkap
    permintaan: list[dict] = field(default_factory=list)     # satu entri per permintaan API (untuk batch_llm)


def _tulis_jejak(folder: Path, entri: dict, system: str, user: str, peta: PetaBatch,
                 resp: LLMResponse | None, galat: str | None, catatan_lengkap: list[str]) -> str:
    """Simpan satu permintaan apa adanya untuk tracing: metadata, prompt, keluaran MENTAH model, peta label
    lokal → id_cpmk (label MKxx diacak per batch), dan traceback bila gagal. Isinya hanya teks kurikulum."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{entri['id_batch']}-p{entri['percobaan']}.json"
    isi = {"entri": entri, "galat": galat, "catatan_validasi": catatan_lengkap,
           "keluaran_mentah": resp.teks if resp else None,
           "ringkasan_thinking": resp.ringkasan_thinking if resp else None,
           "peta_label": {lb: {"kode_mk": k, "id_cpmk": i} for lb, (k, i) in peta.cpmk.items()},
           "prompt": {"system": system, "user": user}}
    path.write_text(json.dumps(isi, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return str(path)


def nilai_batch(client: LLMClient, mks: list[KonteksMK], run: int, cfg: Config, templat: TemplatPrompt,
                heur: dict[str, tuple[str, str, str]], id_batch: str, jejak: Path | None = None) -> HasilBatch:
    """Satu batch (banyak MK) untuk satu pengulangan. Hasil dipisah per MK; MK yang tidak lengkap dikirim ulang
    sendiri (≤ ``llm.max_retry``); bila keluaran terpotong (batas token), batch dibelah dua dan diproses
    terpisah. Galat konfigurasi (kunci/model) langsung dilempar. Setiap permintaan dicatat real-time ke
    ``logs/app.log`` dan, bila ``jejak`` diberikan, disimpan utuh (prompt + keluaran mentah) di folder itu."""
    log = logging.getLogger("crs.llm")
    hb = HasilBatch()
    sisa = list(mks)
    for percobaan in range(1, cfg.llm.max_retry + 2):
        if not sisa:
            break
        rng = rng_batch(cfg.seed, run, f"{id_batch}#{percobaan}")
        system, user, peta = bangun_prompt_batch(sisa, rng, templat)
        if isinstance(client, MockClient):
            client.penjawab = penjawab_mock_bloom(heur, peta)
        entri = {"id_batch": id_batch, "penilai": str(run), "percobaan": percobaan, "n_mk": len(sisa),
                 "n_cpmk": len(peta.cpmk), "timestamp": datetime.now().isoformat(timespec="seconds")}
        log.info("kirim %s percobaan %d: %d MK, %d CPMK, %d karakter prompt (%s)", id_batch, percobaan,
                 len(sisa), len(peta.cpmk), len(system) + len(user), client.model)
        terpotong, resp, galat, cat = False, None, None, []
        t0 = time.perf_counter()
        try:
            resp = client.complete(system, user, skema_penilaian(list(peta.cpmk)))
            terpotong = resp.terpotong
            entri.update(model_pelayan=resp.model, token_masuk=resp.token_masuk, token_keluar=resp.token_keluar,
                         temperature=resp.temperature, thinking=int(resp.thinking), id_respons=resp.id_respons)
            diterima, cat = pisah_per_mk(parse_json_keluaran(resp.teks), peta)
            entri["catatan"] = " | ".join(dict.fromkeys(resp.catatan + cat[:5]))
        except LLMKonfigurasiError:
            log.exception("%s percobaan %d: konfigurasi LLM salah (dihentikan)", id_batch, percobaan)
            raise
        except (LLMError, ValueError) as e:
            diterima, galat = {}, traceback.format_exc()
            entri["catatan"] = (str(e).splitlines()[0][:300] if str(e) else type(e).__name__)
        entri["durasi_detik"] = round(time.perf_counter() - t0, 1)
        for kode, p in diterima.items():
            hb.penilaian[kode] = [q.model_copy(update={"id_cpmk": peta.cpmk[q.id_cpmk][1]}) for q in p]
        sisa = [m for m in sisa if m.kode not in hb.penilaian]
        entri.update(n_mk_diterima=len(diterima), status="ok" if not sisa else "sebagian")
        if jejak is not None:
            entri["berkas_jejak"] = _tulis_jejak(jejak, entri, system, user, peta, resp, galat, cat)
        (log.info if not sisa else log.warning)(
            "selesai %s percobaan %d: %s, %d/%d MK diterima, %.1f dtk, token %s/%s, id_respons %s%s | %s",
            id_batch, percobaan, entri["status"], len(diterima), entri["n_mk"], entri["durasi_detik"],
            entri.get("token_masuk"), entri.get("token_keluar"), entri.get("id_respons"),
            ", TERPOTONG" if terpotong else "", entri.get("catatan", ""))
        if galat:
            log.warning("galat %s percobaan %d:\n%s", id_batch, percobaan, galat)
        hb.permintaan.append(entri)
        if sisa and terpotong and len(sisa) > 1:
            # keluaran terpotong: belah dua; tiap belahan mendapat jatah retry sendiri
            log.warning("%s terpotong batas token → dibelah dua (%d MK tersisa)", id_batch, len(sisa))
            tengah = len(sisa) // 2
            for i, bagian in enumerate((sisa[:tengah], sisa[tengah:]), start=1):
                sub = nilai_batch(client, bagian, run, cfg, templat, heur, f"{id_batch}.{i}", jejak)
                hb.penilaian.update(sub.penilaian)
                hb.permintaan += sub.permintaan
            sisa = [m for m in sisa if m.kode not in hb.penilaian]
            break
    hb.gagal = [m.kode for m in sisa]
    return hb


def baris_rating(client: LLMClient, hb: HasilBatch, mks: list[KonteksMK], run: int, templat: TemplatPrompt,
                 id_batch: str) -> list[dict]:
    """Baris ``rating_bloom`` per CPMK dari hasil batch (MK gagal → ``perlu_tinjau_manual`` tanpa label)."""
    pertama = hb.permintaan[0] if hb.permintaan else {}
    dasar = {"penilai": str(run), "sumber": "llm_mock" if client.provider == "mock" else "llm",
             "model": client.model, "versi_prompt": templat.id, "provider": client.provider,
             "model_pelayan": pertama.get("model_pelayan") or client.model, "temperature": pertama.get("temperature"),
             "thinking": pertama.get("thinking"), "timestamp": datetime.now().isoformat(timespec="seconds")}
    rows = []
    for k in mks:
        if k.kode in hb.penilaian:
            rows += [{**dasar, "kode_mk": k.kode, "id_cpmk": p.id_cpmk, "bloom": p.bloom, "ranah": p.ranah,
                      "alasan": p.alasan, "penalaran": p.penalaran, "keyakinan": p.keyakinan, "status": "ok",
                      "percobaan": len(hb.permintaan), "catatan": f"batch {id_batch}"} for p in hb.penilaian[k.kode]]
        else:
            rows += [{**dasar, "kode_mk": k.kode, "id_cpmk": i, "bloom": "", "status": "perlu_tinjau_manual",
                      "percobaan": len(hb.permintaan),
                      "catatan": f"batch {id_batch}: tidak lengkap setelah retry"} for i in k.id_cpmk]
    return rows


# ---------------------------------------------------------------------
# Rating: muat, jalankan LLM, konsensus
# ---------------------------------------------------------------------


def rating_heuristik(sk: SumberKonteks) -> pd.DataFrame:
    df = sk.cpmk
    rows = [{"sumber": "heuristik", "id_cpmk": r.id_cpmk, "kode_mk": r.kode_mk, "penilai": "1",
             "model": "heuristik-kko", "versi_prompt": "18_kko_bloom_map", "bloom": r.bloom_maks_heuristik,
             "ranah": _ranah_heuristik(r.bloom_maks_heuristik, r.ranah_heuristik),
             "alasan": f"KKO: {getattr(r, 'kko_semua', '')}", "status": "ok"} for r in df.itertuples()]
    return pd.DataFrame(rows).reindex(columns=KOLOM_RATING)


def muat_rating(cfg: Config, sumber: str, sk: SumberKonteks | None = None,
                model: str | None = None, versi_prompt: str | None = None) -> pd.DataFrame | None:
    """Rating format panjang (satu baris per CPMK per penilai); None bila belum ada.

    Untuk ``llm``/``llm_mock``, bila ``model``/``versi_prompt`` tidak diberikan dipakai kombinasi
    (model, versi prompt) yang PALING BARU agar hasil beberapa model tidak tercampur.
    """
    if sumber == "heuristik":
        return rating_heuristik(sk or SumberKonteks.muat(cfg))
    if sumber not in SUMBER_TERSIMPAN:
        raise ValueError(f"sumber rating tidak dikenal: {sumber}")
    df = db.baca(cfg, "SELECT * FROM rating_bloom WHERE sumber = ? ORDER BY kode_mk, id_cpmk, penilai", (sumber,))
    if df.empty:
        return None
    if sumber != "pakar":
        if model is None or versi_prompt is None:
            terbaru = df.sort_values("timestamp").iloc[-1]
            model = model or terbaru["model"]
            versi_prompt = versi_prompt or terbaru["versi_prompt"]
        df = df[(df["model"] == model) & (df["versi_prompt"] == versi_prompt)]
        if df.empty:
            return None
    df = df.reindex(columns=KOLOM_RATING).copy()
    df["bloom"] = df["bloom"].fillna("").astype(str)
    return df.reset_index(drop=True)


def impor_pakar(cfg: Config, df: pd.DataFrame, sk: SumberKonteks | None = None) -> dict:
    """UC-10: impor formulir pakar (``id_cpmk, kode_mk, kode_pakar, bloom, catatan``) ke basis data.

    Validasi: kolom wajib, ``id_cpmk`` dikenal dan cocok dengan ``kode_mk``, label C1–C6/A atau kosong
    (= belum dinilai), satu baris per (pakar, CPMK). Data pakar yang sama diganti seluruhnya.
    """
    sk = sk or SumberKonteks.muat(cfg)
    kurang = [c for c in ("id_cpmk", "kode_mk", "kode_pakar", "bloom") if c not in df.columns]
    if kurang:
        raise ValueError(f"kolom wajib tidak ada: {kurang}")
    d = df.copy()
    for c in ("id_cpmk", "kode_mk", "kode_pakar", "bloom"):
        d[c] = d[c].fillna("").astype(str).str.strip()
    d["bloom"] = d["bloom"].str.upper()
    d["catatan"] = d["catatan"].fillna("").astype(str) if "catatan" in d.columns else ""
    if (d["kode_pakar"] == "").any():
        raise ValueError(f"{int((d['kode_pakar'] == '').sum())} baris tanpa kode_pakar")
    peta = dict(zip(sk.cpmk["id_cpmk"], sk.cpmk["kode_mk"]))
    asing = sorted(set(d["id_cpmk"]) - set(peta))
    if asing:
        raise ValueError(f"id_cpmk tidak dikenal: {asing[:5]}{' …' if len(asing) > 5 else ''}")
    salah_mk = d[d["id_cpmk"].map(peta) != d["kode_mk"]]
    if len(salah_mk):
        raise ValueError(f"kode_mk tidak cocok dengan id_cpmk pada {len(salah_mk)} baris "
                         f"(mis. {salah_mk.iloc[0]['id_cpmk']})")
    tidak_sah = sorted(set(d["bloom"]) - set(LEVEL_BLOOM) - {"A", ""})
    if tidak_sah:
        raise ValueError(f"label bloom tidak dikenal {tidak_sah} (C1–C6, A, atau kosong)")
    ganda = d[d.duplicated(["kode_pakar", "id_cpmk"], keep=False)]
    if len(ganda):
        raise ValueError(f"baris ganda (kode_pakar, id_cpmk): {ganda.iloc[0]['kode_pakar']}, "
                         f"{ganda.iloc[0]['id_cpmk']}")
    waktu = datetime.now().isoformat(timespec="seconds")
    rows = [{"sumber": "pakar", "id_cpmk": r.id_cpmk, "kode_mk": r.kode_mk, "penilai": r.kode_pakar,
             "model": "pakar", "versi_prompt": "instrumen-dpa", "bloom": r.bloom,
             "status": "ok" if r.bloom else "belum_dinilai", "catatan": r.catatan, "timestamp": waktu}
            for r in d.itertuples()]
    pakar = sorted(set(d["kode_pakar"]))
    with db.koneksi(cfg) as c:
        c.executemany("DELETE FROM rating_bloom WHERE sumber = 'pakar' AND penilai = ?", [(p,) for p in pakar])
        db.tulis(c, "rating_bloom", rows)
    return {"n_pakar": len(pakar), "pakar": pakar, "n_baris": len(rows),
            "n_dinilai": int((d["bloom"] != "").sum()), "n_cpmk_kurikulum": len(peta)}


def ekspor_rating_llm(cfg: Config) -> pd.DataFrame:
    """Cadangan seluruh rating LLM tersimpan (semua model & versi prompt, sumber llm/llm_mock) — iterasi 3.
    Dipakai untuk memulihkan hasil penilaian berbayar setelah sistem diatur ulang (uji dari nol)."""
    df = db.baca(cfg, "SELECT * FROM rating_bloom WHERE sumber IN ('llm', 'llm_mock') "
                      "ORDER BY model, versi_prompt, penilai, kode_mk, id_cpmk")
    return df.reindex(columns=KOLOM_RATING)


def impor_rating_llm(cfg: Config, df: pd.DataFrame, sk: SumberKonteks | None = None) -> dict:
    """Pulihkan rating LLM dari cadangan CSV (format ``KOLOM_RATING``). Asal-usul (model, versi prompt, waktu)
    diambil dari berkas apa adanya dan dicatat di log proses. Rating dengan (sumber, model, versi prompt) yang
    sama diganti seluruhnya. Validasi: kolom wajib, sumber llm/llm_mock, ``id_cpmk`` dikenal & cocok dengan
    ``kode_mk``, label C1–C6/A untuk status ok, tanpa baris ganda."""
    sk = sk or SumberKonteks.muat(cfg)
    wajib = ["sumber", "id_cpmk", "kode_mk", "penilai", "model", "versi_prompt", "bloom", "status"]
    kurang = [c for c in wajib if c not in df.columns]
    if kurang:
        raise ValueError(f"kolom wajib tidak ada: {kurang}")
    if df.empty:
        raise ValueError("berkas tidak berisi baris rating")
    d = df.reindex(columns=KOLOM_RATING).copy()
    for c in wajib:
        d[c] = d[c].fillna("").astype(str).str.strip()
    d["bloom"] = d["bloom"].str.upper()
    salah = sorted(set(d["sumber"]) - {"llm", "llm_mock"})
    if salah:
        raise ValueError(f"sumber harus llm atau llm_mock, ditemukan {salah}")
    if (d[["model", "versi_prompt", "penilai"]] == "").any().any():
        raise ValueError("kolom model, versi_prompt, dan penilai tidak boleh kosong")
    peta = dict(zip(sk.cpmk["id_cpmk"], sk.cpmk["kode_mk"]))
    asing = sorted(set(d["id_cpmk"]) - set(peta))
    if asing:
        raise ValueError(f"id_cpmk tidak dikenal: {asing[:5]}{' …' if len(asing) > 5 else ''}")
    if (d["id_cpmk"].map(peta) != d["kode_mk"]).any():
        raise ValueError("kode_mk tidak cocok dengan id_cpmk pada sebagian baris")
    ok = d["status"] == "ok"
    tidak_sah = sorted(set(d.loc[ok, "bloom"]) - set(LEVEL_BLOOM) - {"A"})
    if tidak_sah:
        raise ValueError(f"label bloom tidak dikenal pada baris berstatus ok: {tidak_sah} (C1–C6 atau A)")
    kunci = ["sumber", "model", "versi_prompt", "penilai", "id_cpmk"]
    if d.duplicated(kunci).any():
        g = d[d.duplicated(kunci)].iloc[0]
        raise ValueError(f"baris ganda: {g['model']} {g['versi_prompt']} pengulangan {g['penilai']} {g['id_cpmk']}")
    run = d[["sumber", "model", "versi_prompt"]].drop_duplicates().values.tolist()
    with catat_proses(cfg, "impor_rating_llm", {"run": [" · ".join(r) for r in run]}) as log:
        with db.koneksi(cfg) as c:
            c.executemany("DELETE FROM rating_bloom WHERE sumber = ? AND model = ? AND versi_prompt = ?",
                          [tuple(r) for r in run])
            db.tulis(c, "rating_bloom", d.to_dict("records"))
        hasil = {"n_baris": int(len(d)), "n_mk": int(d["kode_mk"].nunique()), "n_cpmk": int(d["id_cpmk"].nunique()),
                 "n_pengulangan": int(d["penilai"].nunique()), "run": [" · ".join(r) for r in run]}
        log.ringkasan.update(hasil)
    return hasil


def konsensus_bloom(rating: pd.DataFrame) -> pd.DataFrame:
    """Satu label per CPMK dari beberapa pengulangan/pakar: mayoritas 'A' → A; selain itu median
    bawah (lower median) level C CPMK. Hanya baris berstatus ok dengan label terisi."""
    df = rating[(rating["status"] == "ok") & rating["bloom"].isin(list(LEVEL_BLOOM) + ["A"])]
    baris = []
    for (id_cpmk, kode), g in df.groupby(["id_cpmk", "kode_mk"], sort=True):
        label = list(g["bloom"])
        n_a = label.count("A")
        if n_a * 2 > len(label):
            hasil = "A"
        else:
            lv = sorted(int(x[1]) for x in label if x != "A")
            hasil = f"C{lv[(len(lv) - 1) // 2]}"
        baris.append({"id_cpmk": id_cpmk, "kode_mk": kode, "bloom": hasil, "n_penilaian": len(label),
                      "sepakat": len(set(label)) == 1})
    return pd.DataFrame(baris, columns=["id_cpmk", "kode_mk", "bloom", "n_penilaian", "sepakat"])


def cl_mk(konsensus: pd.DataFrame, kur: Kurikulum, metode: str, rating: pd.DataFrame | None = None,
          cl_afektif: float | None = None) -> pd.DataFrame:
    """CL per MK = agregasi level C1..C6 CPMK kognitif (konsensus); CPMK 'A' dikecualikan.

    MK yang seluruh CPMK-nya berkonsensus afektif (mis. agama) tetap diberi CL agar tidak jatuh ke nilai
    bawaan: (1) bila ``rating`` diberikan, dari level C yang muncul pada pengulangan/penilai individual MK itu
    (komponen kognitif minoritas, mis. ranah afektif+kognitif); (2) selain itu ``cl_afektif`` (config).
    Keduanya asumsi yang dicatat di kolom ``keterangan``. RM1 memanggil tanpa fallback."""
    fungsi = {"median": lambda v: float(np.median(v)), "maks": lambda v: float(max(v)),
              "rata": lambda v: round(float(np.mean(v)), 4)}[metode]
    ok = None
    if rating is not None:
        ok = rating[(rating["status"] == "ok") & rating["bloom"].isin(LEVEL_BLOOM)]
    baris = []
    for kode in kur.kode:
        label = list(konsensus.loc[konsensus["kode_mk"] == kode, "bloom"])
        lv = [int(x[1]) for x in label if x != "A"]
        n_a = len(label) - len(lv)
        if not label:
            cl, ket = None, "tidak ada CPMK di dokumen sumber (CLAUDE.md §7a)"
        elif not lv:
            individu = [int(x[1]) for x in ok.loc[ok["kode_mk"] == kode, "bloom"]] if ok is not None else []
            if individu:
                cl = fungsi(individu)
                ket = (f"seluruh {n_a} CPMK berkonsensus afektif (A); CL dari {len(individu)} penilaian kognitif "
                       f"individual ({metode}, asumsi)")
            elif cl_afektif is not None:
                cl, ket = float(cl_afektif), f"seluruh {n_a} CPMK afektif (A); CL = beban.cl_afektif {cl_afektif:g} (asumsi)"
            else:
                cl, ket = None, f"seluruh {n_a} CPMK afektif (A), tidak dihitung"
        else:
            cl = fungsi(lv)
            ket = f"{metode} level Bloom {len(lv)} CPMK kognitif ({', '.join(x for x in label if x != 'A')})"
            if n_a:
                ket += f"; {n_a} CPMK afektif tidak dihitung"
        baris.append({"kode_mk": kode, "cl": cl, "n_cpmk": len(label), "n_cpmk_kognitif": len(lv),
                      "n_cpmk_afektif": n_a, "bloom_cpmk": ";".join(label), "metode": metode, "keterangan": ket})
    return pd.DataFrame(baris)


def justifikasi_cpmk(rating: pd.DataFrame, konsensus: pd.DataFrame) -> dict[str, list[str]]:
    """KF-17: per MK, daftar "id_cpmk: label — alasan" dari penilaian yang labelnya sama dengan konsensus
    (alasan dari penilai/pengulangan pertama yang sepakat dengan konsensus)."""
    ok = rating[rating["status"] == "ok"]
    hasil: dict[str, list[str]] = {}
    for r in konsensus.itertuples():
        cocok = ok[(ok["id_cpmk"] == r.id_cpmk) & (ok["bloom"] == r.bloom)]
        alasan = next((str(a) for a in cocok["alasan"] if isinstance(a, str) and a.strip()), "")
        hasil.setdefault(r.kode_mk, []).append(f"{r.id_cpmk}: {r.bloom}" + (f", {alasan}" if alasan else ""))
    return hasil


def _hitung_cl(cfg: Config, kur: Kurikulum | None = None) -> tuple[pd.DataFrame, dict, dict[str, list[str]]]:
    kur = kur or load_kurikulum(cfg)
    sk = SumberKonteks.muat(cfg, kur)
    sumber, catatan = cfg.beban.sumber_cl, None
    rating = muat_rating(cfg, sumber, sk)
    if rating is None or not (rating["status"] == "ok").any():
        catatan = f"rating '{sumber}' belum tersedia → memakai heuristik KKO"
        sumber, rating = "heuristik", muat_rating(cfg, "heuristik", sk)
    konsensus = konsensus_bloom(rating)
    tabel = cl_mk(konsensus, kur, cfg.beban.agregasi_cl, rating, cfg.beban.cl_afektif)
    tabel["sumber"] = sumber
    info = {"sumber": f"Bloom per CPMK ({sumber}) → CL = {cfg.beban.agregasi_cl} level C1–C6 CPMK kognitif",
            "sumber_rating": sumber, "metode": cfg.beban.agregasi_cl, "catatan": catatan,
            "n_mk": int(tabel["cl"].notna().sum())}
    return tabel, info, justifikasi_cpmk(rating, konsensus)


def tabel_cl(cfg: Config, kur: Kurikulum | None = None) -> tuple[pd.DataFrame, dict]:
    """CL per MK menurut ``beban.sumber_cl`` (fallback heuristik bila rating sumber belum ada)."""
    tabel, info, _ = _hitung_cl(cfg, kur)
    return tabel, info


@dataclass
class BebanMK:
    """Masukan beban kognitif untuk planner: CL, keterangan agregasi, info sumber, justifikasi per CPMK."""

    cl: dict[str, float]
    keterangan: dict[str, str]
    info: dict
    justifikasi: dict[str, list[str]]


def muat_beban_mk(cfg: Config, kur: Kurikulum) -> BebanMK:
    tabel, info, just = _hitung_cl(cfg, kur)
    ada = tabel[tabel["cl"].notna()]
    return BebanMK(dict(zip(ada["kode_mk"], ada["cl"].astype(float))),
                   dict(zip(tabel["kode_mk"], tabel["keterangan"])), info, just)


def jalankan_cl(cfg: Config) -> dict:
    """Simpan CL per MK ke tabel ``cl_mk`` (transparansi & ekspor)."""
    tabel, info = tabel_cl(cfg)
    db.ganti_isi(cfg, "cl_mk", tabel)
    db.simpan_ringkasan(cfg, "cl", info)
    return info


def jalankan_penilaian_bloom(cfg: Config, provider: str | None = None, model: str | None = None,
                             kode_list: list[str] | None = None, lanjutkan: bool = True,
                             progress: Callable[[int, int, str], None] | None = None) -> dict:
    """LLM-as-judge Bloom per CPMK secara BATCH: setiap pengulangan (``llm.n_pengulangan``) = satu permintaan
    berisi seluruh MK (atau potongan ``llm.mk_per_batch``), keluaran JSON schema ketat dipisah per MK lalu
    ditulis ke ``rating_bloom`` (penilai = nomor pengulangan). Pengulangan berjalan paralel (``llm.konkurensi``)
    bila klien aman paralel. ``lanjutkan`` melewati (MK, pengulangan) yang sudah berhasil untuk model & versi
    prompt yang sama; ``lanjutkan=False`` mengulang semuanya."""
    client = buat_client(cfg, provider, model)
    sumber = "llm_mock" if client.provider == "mock" else "llm"
    templat = muat_prompt("bloom", cfg.llm.versi_prompt_bloom)
    sk = SumberKonteks.muat(cfg)
    heur = peta_heuristik(sk)
    kode_list = kode_list or sk.kode_dengan_cpmk()
    kunci = (sumber, client.model, templat.id)
    if not lanjutkan:
        with db.koneksi(cfg) as c:
            c.executemany("DELETE FROM rating_bloom WHERE sumber = ? AND model = ? AND versi_prompt = ? "
                          "AND kode_mk = ?", [(*kunci, k) for k in kode_list])
    lama = db.baca(cfg, "SELECT DISTINCT kode_mk, penilai FROM rating_bloom WHERE sumber = ? AND model = ? "
                        "AND versi_prompt = ? AND status = 'ok'", kunci)
    selesai = set(zip(lama["kode_mk"], lama["penilai"].astype(int)))
    ukuran = cfg.llm.mk_per_batch or len(kode_list)
    tugas = []                                     # (run, id_batch, [KonteksMK])
    for run in range(1, cfg.llm.n_pengulangan + 1):
        sisa = [k for k in kode_list if (k, run) not in selesai]
        for i in range(0, len(sisa), ukuran):
            tugas.append((run, f"r{run}-b{i // ukuran + 1}", [sk.mk(k) for k in sisa[i:i + ukuran]]))
    konkurensi = cfg.llm.konkurensi if client.aman_paralel else 1
    stempel = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"   # unik per pemanggilan
    jejak = cfg.logs_dir / "llm" / f"bloom-{stempel}"     # prompt + keluaran mentah per permintaan (tracing)
    teruskan_log_sdk(cfg)                                  # retry SDK (429/5xx/timeout) & status HTTP → app.log
    param = {"provider": client.provider, "model": client.model, "versi_prompt": templat.id,
             "n_pengulangan": cfg.llm.n_pengulangan, "mk_per_batch": cfg.llm.mk_per_batch, "effort": cfg.llm.effort,
             "konkurensi": konkurensi, "n_mk": len(kode_list), "folder_jejak": str(jejak),
             "max_tokens": cfg.llm.max_tokens, "timeout_detik": cfg.llm.timeout_detik}
    logging.getLogger("crs.llm").info("penilaian Bloom mulai: %s", json.dumps(param, ensure_ascii=False))
    with catat_proses(cfg, "penilaian_bloom", param) as log:
        hitung = {"permintaan": 0, "mk_baru": 0, "gagal": 0, "token_masuk": 0, "token_keluar": 0}

        def simpan(run: int, id_batch: str, mks: list[KonteksMK], hb: HasilBatch) -> None:
            rows = baris_rating(client, hb, mks, run, templat, id_batch)
            with db.koneksi(cfg) as c:   # satu transaksi per batch → dapat dilanjutkan bila terputus
                c.executemany("DELETE FROM rating_bloom WHERE sumber = ? AND model = ? AND versi_prompt = ? "
                              "AND kode_mk = ? AND penilai = ?", [(*kunci, k.kode, str(run)) for k in mks])
                db.tulis(c, "rating_bloom", rows)
                db.tulis(c, "batch_llm", [{**e, "id_batch": f"{stempel}-{e['id_batch']}", "sumber": sumber,
                                           "model": client.model, "versi_prompt": templat.id}
                                          for e in hb.permintaan])
            hitung["permintaan"] += len(hb.permintaan)
            hitung["mk_baru"] += len(hb.penilaian)
            hitung["gagal"] += len(hb.gagal)
            hitung["token_masuk"] += sum(e.get("token_masuk") or 0 for e in hb.permintaan)
            hitung["token_keluar"] += sum(e.get("token_keluar") or 0 for e in hb.permintaan)

        def kerjakan(t):
            run, id_batch, mks = t
            return nilai_batch(client, mks, run, cfg, templat, heur, id_batch, jejak)

        if konkurensi == 1:
            for i, t in enumerate(tugas, start=1):
                if progress:
                    progress(i, len(tugas), t[1])
                simpan(t[0], t[1], t[2], kerjakan(t))
        elif tugas:
            # hanya panggilan API yang paralel; penulisan basis data tetap di thread ini. Galat satu batch
            # tidak membuang hasil batch lain yang sudah dibayar: semuanya disimpan dulu, galat dilempar di akhir.
            galat_pertama = None
            with ThreadPoolExecutor(max_workers=konkurensi) as ex:
                futur = {ex.submit(kerjakan, t): t for t in tugas}
                for i, f in enumerate(as_completed(futur), start=1):
                    run, id_batch, mks = futur[f]
                    try:
                        simpan(run, id_batch, mks, f.result())
                    except Exception as e:
                        logging.getLogger("crs.llm").exception("batch %s gagal total", id_batch)
                        galat_pertama = galat_pertama or e
                    if progress:
                        progress(i, len(tugas), id_batch)
            if galat_pertama is not None:
                raise galat_pertama
        ringkasan = {"sumber": sumber, "n_batch": len(tugas), "n_permintaan_api": hitung["permintaan"],
                     "n_mk_dinilai": hitung["mk_baru"], "n_dilewati_resume": len(selesai & {
                         (k, r) for k in kode_list for r in range(1, cfg.llm.n_pengulangan + 1)}),
                     "n_perlu_tinjau_manual": hitung["gagal"], "token_masuk": hitung["token_masuk"],
                     "token_keluar": hitung["token_keluar"], **param}
        log.ringkasan.update(ringkasan)
    return ringkasan


def konsistensi_pengulangan(rating: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Analisis antarpengulangan (batch) per CPMK: label tiap pengulangan, sepakat penuh, rentang level C,
    dan label konsensus. Mengembalikan (tabel per CPMK, ringkasan)."""
    ok = rating[(rating["status"] == "ok") & (rating["bloom"] != "")]
    if ok.empty:
        return pd.DataFrame(), {}
    lebar = ok.pivot_table(index=["kode_mk", "id_cpmk"], columns="penilai", values="bloom", aggfunc="first")
    lebar.columns = [f"pengulangan_{c}" for c in lebar.columns]
    kons = konsensus_bloom(rating).set_index(["kode_mk", "id_cpmk"])
    baris = []
    for (kode, idc), r in lebar.iterrows():
        label = [x for x in r.tolist() if isinstance(x, str)]
        lv = [int(x[1]) for x in label if x != "A"]
        baris.append({"kode_mk": kode, "id_cpmk": idc, **r.to_dict(), "konsensus": kons.loc[(kode, idc), "bloom"],
                      "n_pengulangan": len(label), "sepakat_penuh": len(set(label)) == 1,
                      "campur_afektif": 0 < label.count("A") < len(label),
                      "rentang_level": (max(lv) - min(lv)) if len(lv) > 1 else 0})
    df = pd.DataFrame(baris)
    lengkap = df[df["n_pengulangan"] > 1]
    ringkas = {"n_cpmk": int(len(df)),
               "proporsi_sepakat_penuh": round(float(lengkap["sepakat_penuh"].mean()), 4) if len(lengkap) else None,
               "proporsi_rentang_le1": round(float((lengkap["rentang_level"] <= 1).mean()), 4) if len(lengkap) else None,
               "n_campur_afektif": int(lengkap["campur_afektif"].sum()),
               "n_mk_semua_sepakat": int(lengkap.groupby("kode_mk")["sepakat_penuh"].all().sum())}
    return df, ringkas


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = load_config()
    print(json.dumps(jalankan_penilaian_bloom(c, lanjutkan="--baru" not in sys.argv,
                                              progress=lambda i, n, k: print(f"[{i}/{n}] {k}")),
                     indent=2, ensure_ascii=False))
    print(json.dumps(jalankan_cl(c), indent=2, ensure_ascii=False))
