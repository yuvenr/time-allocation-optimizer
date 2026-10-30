"""Penyimpanan data operasional dan hasil proses dengan dua *backend* berskema sama:

- **SQLite** (bawaan; ``paths.db``, default ``data/crs.db``) untuk pengembangan lokal dan tes;
- **PostgreSQL** bila ``paths.db_url`` / variabel lingkungan ``DATABASE_URL`` berisi ``postgresql://...``
  — dipakai saat aplikasi di-*hosting* (Streamlit Community Cloud / Azure) karena sistem berkas
  server tersebut tidak persisten, sehingga berkas SQLite akan hilang setiap kali aplikasi dimulai ulang.

Seluruh modul lain hanya memanggil fungsi di berkas ini (``koneksi``, ``tulis``, ``baca``, ...) dengan
placeholder ``?``; penerjemahan dialek (placeholder ``%s``, ``INSERT OR REPLACE`` → ``ON CONFLICT``,
``REAL`` → ``DOUBLE PRECISION``, katalog tabel) dilakukan di sini.

Yang disimpan di basis data:
- data akademik nyata yang sudah dianonimkan (``krs_asli``: KRS satu semester; ``ip_asli``: IP/IPK
  semester sebelumnya) — tanpa NIM (``src/akademik.py``);
- data operasional: histori KRS yang sudah dianonimkan, transaksi latih/uji, aturan asosiasi,
  support item (masing-masing dengan kolom ``sumber``: asli | sintetis | fixture), rating Bloom per
  CPMK (LLM, mock, pakar; termasuk pemakaian token), skor CL per MK, rencana studi beserta catatan dan
  validasi DPA, override kurikulum, log proses (KNF-09), serta log akses peran (``log_akses``);
- ringkasan proses (JSON) di tabel ``ringkasan``;
- tabel hasil analisis berkolom dinamis (sensitivitas, simulasi, evaluasi) dengan prefiks ``hasil_``.

Yang TIDAK disimpan di sini:
- dataset Kurikulum 2024 (``data/kurikulum_2024``) tetap berupa CSV hasil build yang dibaca
  apa adanya (CLAUDE.md §2), sehingga dapat diregenerasi dan dilacak di git;
- berkas KRS mentah (berisi NIM) hanya menjadi masukan impor; yang masuk ke basis data hanya
  hasil anonimisasi (KNF-04);
- ekspor CSV untuk JASP ditulis ke ``paths.ekspor``.

Skema diberi versi (``meta.versi_skema``) dan dibuat otomatis saat koneksi pertama.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import sqlite3
import threading
import time
import weakref
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from src.config import Config

VERSI_SKEMA = 6

# Migrasi berurutan: versi tujuan → perintah SQL (dijalankan sekali pada basis data versi lebih lama).
# Tabel turunan yang dihapus di sini dibuat ulang oleh DDL terbaru dan diisi ulang oleh pipeline.
MIGRASI: dict[int, list[str]] = {
    2: ["ALTER TABLE rating_bloom ADD COLUMN ringkasan_thinking TEXT"],   # ringkasan penalaran model per permintaan
    3: ["ALTER TABLE rating_bloom ADD COLUMN token_masuk INTEGER",        # biaya per permintaan LLM
        "ALTER TABLE rating_bloom ADD COLUMN token_keluar INTEGER",
        # v2: kolom asal data (sumber) + transaksi KRS nyata tanpa IP semester berjalan
        "DROP TABLE IF EXISTS histori", "DROP TABLE IF EXISTS transaksi", "DROP TABLE IF EXISTS aturan",
        "DROP TABLE IF EXISTS support_item",
        "DELETE FROM ringkasan WHERE nama IN ('preprocessing', 'arm', 'theta', 'simulasi', 'pipeline')"],
    4: [],   # tabel batch_llm (dibuat oleh DDL): jejak permintaan penilaian Bloom per batch
    5: ["ALTER TABLE batch_llm ADD COLUMN id_respons TEXT",           # tracing: ID respons penyedia,
        "ALTER TABLE batch_llm ADD COLUMN durasi_detik REAL",         # durasi per permintaan,
        "ALTER TABLE batch_llm ADD COLUMN berkas_jejak TEXT"],        # berkas prompt+keluaran mentah (logs/llm/)
    6: [],   # tabel berkas (dibuat oleh DDL): salinan berkas raw & parameter UI (hosting tanpa disk persisten)
}

DDL = """
CREATE TABLE IF NOT EXISTS meta (
    kunci TEXT PRIMARY KEY,
    nilai TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ringkasan (
    nama TEXT PRIMARY KEY,
    isi TEXT NOT NULL,                       -- JSON
    diperbarui TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS log_proses (
    run_id TEXT PRIMARY KEY,
    proses TEXT NOT NULL,
    mulai TEXT NOT NULL,
    selesai TEXT NOT NULL,
    durasi_detik REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    seed INTEGER,
    data_krs TEXT,
    parameter TEXT,                          -- JSON
    ringkasan TEXT,                          -- JSON
    galat TEXT
);
CREATE INDEX IF NOT EXISTS ix_log_proses ON log_proses (proses, mulai);

-- Data akademik nyata yang sudah dianonimkan (KD-04, KD-10; src/akademik.py) — tanpa NIM
CREATE TABLE IF NOT EXISTS krs_asli (
    id_anon TEXT NOT NULL,
    angkatan INTEGER NOT NULL,
    semester_ke INTEGER NOT NULL CHECK (semester_ke >= 1),
    periode TEXT NOT NULL,
    kode_mk TEXT NOT NULL,
    nama_mk TEXT,
    sks INTEGER NOT NULL,
    kurikulum_kode TEXT,
    PRIMARY KEY (id_anon, periode, kode_mk)
);
CREATE TABLE IF NOT EXISTS ip_asli (
    id_anon TEXT NOT NULL,
    angkatan INTEGER NOT NULL,
    semester_ke INTEGER NOT NULL CHECK (semester_ke >= 1),
    periode TEXT NOT NULL,
    ip_beban REAL, sks_beban INTEGER, ipk_beban REAL, sksk_beban INTEGER,
    ip_lulus REAL, sks_lulus INTEGER, ipk_lulus REAL, sksk_lulus INTEGER,
    sks_tidak_lulus INTEGER,
    PRIMARY KEY (id_anon, periode)
);

-- Histori KRS multisemester (bernilai) setelah anonimisasi + normalisasi ke Kurikulum 2024 (UC-07; KD-04)
CREATE TABLE IF NOT EXISTS histori (
    id_anon TEXT NOT NULL,
    angkatan INTEGER NOT NULL,
    semester_ke INTEGER NOT NULL CHECK (semester_ke >= 1),
    kode_mk TEXT NOT NULL,
    kode_asal TEXT NOT NULL,
    jenis_perubahan TEXT NOT NULL,
    sks INTEGER NOT NULL,
    sks_asal INTEGER,
    nilai_huruf TEXT NOT NULL,
    lulus INTEGER NOT NULL CHECK (lulus IN (0, 1)),
    ip_semester REAL NOT NULL,
    kurikulum TEXT,
    masa_studi_semester INTEGER NOT NULL,
    sumber TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('latih', 'uji'))
);
CREATE INDEX IF NOT EXISTS ix_histori_alumni ON histori (id_anon, semester_ke);

-- Transaksi = MK yang diambil satu mahasiswa pada satu semester (KF-05); gagal = yang tidak lulus
CREATE TABLE IF NOT EXISTS transaksi (
    sumber TEXT NOT NULL,                    -- asli (KRS akademik) | sintetis | fixture
    split TEXT NOT NULL CHECK (split IN ('latih', 'uji')),
    id_anon TEXT NOT NULL,
    angkatan INTEGER NOT NULL,
    semester_ke INTEGER NOT NULL,
    ip_semester REAL,                        -- IP semester itu (kosong untuk KRS berjalan)
    items TEXT NOT NULL,                     -- JSON list "kode@S"
    gagal TEXT NOT NULL,                     -- JSON list kode
    PRIMARY KEY (id_anon, semester_ke)
);
CREATE INDEX IF NOT EXISTS ix_transaksi_split ON transaksi (split, sumber);

-- Aturan asosiasi (KD-09) dan support item tunggal; sumber = asal transaksi kelompok semester itu
CREATE TABLE IF NOT EXISTS aturan (
    anteseden TEXT NOT NULL,
    konsekuen TEXT NOT NULL,
    support REAL NOT NULL,
    confidence REAL NOT NULL,
    lift REAL NOT NULL,
    sumber TEXT NOT NULL,
    run_id TEXT NOT NULL,
    PRIMARY KEY (anteseden, konsekuen)
);
CREATE TABLE IF NOT EXISTS support_item (
    item TEXT PRIMARY KEY,
    kode_mk TEXT NOT NULL,
    semester INTEGER NOT NULL,
    support REAL NOT NULL,
    jumlah INTEGER NOT NULL,
    n_transaksi INTEGER NOT NULL,
    sumber TEXT NOT NULL
);

-- Rating Bloom per CPMK (KD-07, KD-08): satu baris per CPMK per pengulangan (LLM) atau per pakar
CREATE TABLE IF NOT EXISTS rating_bloom (
    sumber TEXT NOT NULL CHECK (sumber IN ('llm', 'llm_mock', 'pakar')),
    id_cpmk TEXT NOT NULL,
    kode_mk TEXT NOT NULL,
    penilai TEXT NOT NULL,                   -- nomor pengulangan (LLM) atau kode pakar
    model TEXT NOT NULL DEFAULT '',
    versi_prompt TEXT NOT NULL DEFAULT '',
    bloom TEXT NOT NULL DEFAULT '',
    ranah TEXT,
    penalaran TEXT,
    alasan TEXT,
    keyakinan REAL,
    provider TEXT,
    model_pelayan TEXT,
    temperature REAL,
    thinking INTEGER,
    ringkasan_thinking TEXT,
    token_masuk INTEGER,
    token_keluar INTEGER,
    status TEXT NOT NULL,
    percobaan INTEGER,
    catatan TEXT,
    timestamp TEXT,
    PRIMARY KEY (sumber, model, versi_prompt, penilai, id_cpmk)
);
CREATE INDEX IF NOT EXISTS ix_rating_mk ON rating_bloom (sumber, kode_mk);

-- Satu baris per permintaan API penilaian Bloom berbatch (prompt v3): token, status, percobaan
CREATE TABLE IF NOT EXISTS batch_llm (
    id_batch TEXT NOT NULL,
    sumber TEXT NOT NULL,
    model TEXT NOT NULL,
    versi_prompt TEXT NOT NULL,
    penilai TEXT NOT NULL,                   -- nomor pengulangan
    percobaan INTEGER NOT NULL,
    n_mk INTEGER NOT NULL,
    n_cpmk INTEGER NOT NULL,
    n_mk_diterima INTEGER,
    status TEXT NOT NULL,                    -- ok | sebagian
    model_pelayan TEXT,
    temperature REAL,
    thinking INTEGER,
    token_masuk INTEGER,
    token_keluar INTEGER,
    catatan TEXT,
    timestamp TEXT NOT NULL,
    id_respons TEXT,                         -- ID respons penyedia (tracing ke dasbor/dukungan penyedia)
    durasi_detik REAL,
    berkas_jejak TEXT,                       -- logs/llm/<run>/<batch>-p<percobaan>.json: prompt + keluaran mentah
    PRIMARY KEY (id_batch, percobaan)
);

CREATE TABLE IF NOT EXISTS cl_mk (
    kode_mk TEXT PRIMARY KEY,
    cl REAL,
    n_cpmk INTEGER NOT NULL,
    n_cpmk_kognitif INTEGER NOT NULL,
    n_cpmk_afektif INTEGER NOT NULL,
    bloom_cpmk TEXT,
    metode TEXT NOT NULL,
    keterangan TEXT,
    sumber TEXT NOT NULL
);

-- Rencana studi (KD-11) + catatan & validasi DPA (KD-12, KF-19)
CREATE TABLE IF NOT EXISTS rencana (
    id TEXT PRIMARY KEY,
    dibuat TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('arm', 'hybrid')),
    label TEXT NOT NULL DEFAULT '',
    isi TEXT NOT NULL,                       -- JSON Plan
    status_dpa TEXT NOT NULL DEFAULT 'belum_ditinjau'
        CHECK (status_dpa IN ('belum_ditinjau', 'disetujui', 'perlu_revisi')),
    catatan_dpa TEXT NOT NULL DEFAULT '',
    ditinjau_pada TEXT
);

-- Override kurikulum oleh administrator (KF-01..KF-03; dataset tetap utuh, setiap perubahan tercatat)
CREATE TABLE IF NOT EXISTS override_cpmk (
    id_cpmk TEXT PRIMARY KEY,
    teks TEXT NOT NULL,
    alasan TEXT NOT NULL,
    diperbarui TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS override_prasyarat (
    kode_mk TEXT NOT NULL,
    kode_mk_syarat TEXT NOT NULL,
    aksi TEXT NOT NULL CHECK (aksi IN ('tambah', 'hapus')),
    alasan TEXT NOT NULL,
    diperbarui TEXT NOT NULL,
    PRIMARY KEY (kode_mk, kode_mk_syarat)
);

-- Salinan berkas yang ditulis aplikasi (riwayat KRS di paths.raw, config.override.yaml). Hosting seperti
-- Streamlit Community Cloud tidak punya disk persisten: berkas dipulihkan dari tabel ini saat aplikasi mulai
-- (src/berkas.py). Isi disimpan byte per byte karena sidik data (sidik_raw) di-hash per byte.
CREATE TABLE IF NOT EXISTS berkas (
    nama TEXT PRIMARY KEY,                   -- raw/<nama berkas> | config.override.yaml
    isi BLOB NOT NULL,
    diperbarui TEXT NOT NULL
);

-- Audit masuk peran berproteksi (Administrator, DPA; src/auth.py)
CREATE TABLE IF NOT EXISTS log_akses (
    waktu TEXT NOT NULL,
    peran TEXT NOT NULL,
    berhasil INTEGER NOT NULL CHECK (berhasil IN (0, 1)),
    keterangan TEXT
);
"""


def path_db(cfg: Config) -> Path:
    return cfg.db_path


def sekarang() -> str:
    return datetime.now().isoformat(timespec="seconds")


def backend(cfg: Config) -> str:
    """``postgres`` bila ``paths.db_url`` berisi URL PostgreSQL, selain itu ``sqlite``."""
    url = (cfg.paths.db_url or "").strip()
    return "postgres" if url.startswith(("postgresql://", "postgres://")) else "sqlite"


def ada_db(cfg: Config) -> bool:
    """Basis data sudah ada? (PostgreSQL dianggap selalu ada; skema dibuat saat koneksi pertama)."""
    return backend(cfg) == "postgres" or path_db(cfg).exists()


def keterangan(cfg: Config) -> str:
    """Label lokasi basis data untuk UI/log tanpa membocorkan kata sandi pada URL."""
    if backend(cfg) == "sqlite":
        return f"SQLite · {path_db(cfg)}"
    from urllib.parse import urlsplit
    u = urlsplit(cfg.paths.db_url)
    skema = f" · skema {cfg.paths.db_schema}" if cfg.paths.db_schema else ""
    return f"PostgreSQL · {u.hostname or 'localhost'}{u.path}{skema}"


@contextmanager
def koneksi(cfg: Config) -> Iterator[Any]:
    """Koneksi transaksional: commit bila sukses, rollback bila gagal. Skema dibuat otomatis."""
    if backend(cfg) == "postgres":
        with _koneksi_pg(cfg) as conn:
            yield conn
        return
    path = path_db(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")      # pembaca (sesi UI) tidak terkunci saat proses panjang menulis
        _siapkan(conn)
        conn.commit()          # skema/migrasi tidak ikut dibatalkan bila transaksi pemanggil gagal
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _siapkan(conn: sqlite3.Connection) -> None:
    baru = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'meta'").fetchone() is None
    conn.executescript(DDL)
    baris = conn.execute("SELECT nilai FROM meta WHERE kunci = 'versi_skema'").fetchone()
    if baris is None:
        if not baru:
            raise RuntimeError("Basis data tanpa versi skema, bukan dibuat oleh aplikasi ini")
        conn.execute("INSERT INTO meta (kunci, nilai) VALUES ('versi_skema', ?)", (str(VERSI_SKEMA),))
        return
    versi = int(baris["nilai"])
    if versi > VERSI_SKEMA:
        raise RuntimeError(f"Basis data berskema versi {versi} lebih baru dari kode ({VERSI_SKEMA})")
    for v in range(versi + 1, VERSI_SKEMA + 1):          # DDL di atas sudah memakai skema terbaru untuk tabel baru
        for sql in MIGRASI.get(v, []):
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:        # kolom sudah ada (tabel dibuat oleh DDL terbaru)
                if "duplicate column" not in str(e).lower():
                    raise
        conn.execute("UPDATE meta SET nilai = ? WHERE kunci = 'versi_skema'", (str(v),))
    if versi < VERSI_SKEMA:
        conn.executescript(DDL)                          # buat ulang tabel turunan yang dihapus migrasi


def _json(o: Any) -> str:
    def default(x):
        if isinstance(x, (set, frozenset)):
            return sorted(x)
        if isinstance(x, (datetime, Path)):
            return str(x)
        if hasattr(x, "item"):          # skalar numpy
            return x.item()
        return str(x)
    return json.dumps(o, ensure_ascii=False, default=default)


# ---------------------------------------------------------------------
# Ringkasan proses (JSON)
# ---------------------------------------------------------------------


def simpan_ringkasan(cfg: Config, nama: str, data: dict) -> None:
    with koneksi(cfg) as c:
        c.execute("INSERT INTO ringkasan (nama, isi, diperbarui) VALUES (?, ?, ?) "
                  "ON CONFLICT(nama) DO UPDATE SET isi = excluded.isi, diperbarui = excluded.diperbarui",
                  (nama, _json(data), sekarang()))


def baca_ringkasan(cfg: Config, nama: str) -> dict | None:
    if not ada_db(cfg):
        return None
    with koneksi(cfg) as c:
        r = c.execute("SELECT isi FROM ringkasan WHERE nama = ?", (nama,)).fetchone()
    return json.loads(r["isi"]) if r else None


def hapus_ringkasan(cfg: Config, nama: str) -> None:
    with koneksi(cfg) as c:
        c.execute("DELETE FROM ringkasan WHERE nama = ?", (nama,))


# ---------------------------------------------------------------------
# Tabel berskema tetap
# ---------------------------------------------------------------------


def kolom_tabel(conn: Any, tabel: str) -> list[str]:
    if isinstance(conn, KoneksiPG):
        return conn.kolom(tabel)
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({tabel})")]


def _nilai(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, bool):              # kolom 0/1 (SQLite menyimpan bool sebagai 1/0; PostgreSQL menolak bool→integer)
        return int(v)
    if isinstance(v, (list, dict, set, frozenset)):
        return _json(v)
    return v


def tulis(conn: Any, tabel: str, baris: Iterable[dict], *, ganti: bool = False) -> int:
    """Sisipkan baris (dict) ke tabel berskema tetap; kolom di luar skema ditolak agar salah ketik ketahuan.
    ``ganti=True`` memakai INSERT OR REPLACE (upsert berdasarkan primary key)."""
    kolom = kolom_tabel(conn, tabel)
    if not kolom:
        raise ValueError(f"tabel {tabel!r} tidak ada")

    def sql_untuk(k: tuple[str, ...]) -> str:
        if ganti and isinstance(conn, KoneksiPG):
            return conn.sql_upsert(tabel, list(k), kolom)
        perintah = "INSERT OR REPLACE" if ganti else "INSERT"
        return f"{perintah} INTO {tabel} ({', '.join(k)}) VALUES ({', '.join('?' * len(k))})"

    # Baris berurutan dengan kolom yang sama dikirim sekaligus (executemany; psycopg 3 memakai mode pipeline),
    # bukan satu perjalanan jaringan per baris: penting untuk PostgreSQL cloud yang jauh (mis. migrasi ke Neon).
    n = 0
    kunci: tuple[str, ...] | None = None
    tumpukan: list[list[Any]] = []

    def kirim() -> None:
        if tumpukan:
            conn.executemany(sql_untuk(kunci), tumpukan)
            tumpukan.clear()

    for b in baris:
        asing = set(b) - set(kolom)
        if asing:
            raise ValueError(f"{tabel}: kolom tidak dikenal {sorted(asing)}")
        k = tuple(b)
        if k != kunci or len(tumpukan) >= 1000:
            kirim()
            kunci = k
        tumpukan.append([_nilai(b[x]) for x in k])
        n += 1
    kirim()
    return n


def ganti_isi(cfg: Config, tabel: str, df: pd.DataFrame, where: str = "", params: tuple = ()) -> int:
    """Hapus baris (seluruhnya atau sesuai ``where``) lalu isi ulang dari DataFrame — satu transaksi."""
    with koneksi(cfg) as c:
        c.execute(f"DELETE FROM {tabel}" + (f" WHERE {where}" if where else ""), params)
        return tulis(c, tabel, df.to_dict("records"))


def baca(cfg: Config, sql: str, params: tuple = ()) -> pd.DataFrame:
    """SELECT → DataFrame (basis data & skema dibuat bila belum ada)."""
    with koneksi(cfg) as c:
        if isinstance(c, KoneksiPG):
            return c.baca_df(sql, params)
        return pd.read_sql_query(sql, c, params=params)


def jumlah(cfg: Config, tabel: str, where: str = "", params: tuple = ()) -> int:
    if not ada_db(cfg):
        return 0
    with koneksi(cfg) as c:
        return int(c.execute(f"SELECT COUNT(*) FROM {tabel}" + (f" WHERE {where}" if where else ""),
                             params).fetchone()[0])


# ---------------------------------------------------------------------
# Tabel hasil analisis (kolom dinamis, prefiks ``hasil_``)
# ---------------------------------------------------------------------


def _nama_hasil(nama: str) -> str:
    if not nama.replace("_", "").isalnum():
        raise ValueError(f"nama tabel hasil tidak sah: {nama!r}")
    return f"hasil_{nama}"


def simpan_hasil(cfg: Config, nama: str, df: pd.DataFrame) -> None:
    with koneksi(cfg) as c:
        if isinstance(c, KoneksiPG):
            c.ganti_tabel_df(_nama_hasil(nama), df)
        else:
            df.to_sql(_nama_hasil(nama), c, if_exists="replace", index=False)


def baca_hasil(cfg: Config, nama: str) -> pd.DataFrame | None:
    if not ada_db(cfg):
        return None
    t = _nama_hasil(nama)
    with koneksi(cfg) as c:
        if isinstance(c, KoneksiPG):
            return c.baca_df(f"SELECT * FROM {t}") if c.ada_tabel(t) else None
        ada = c.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (t,)).fetchone()
        return pd.read_sql_query(f"SELECT * FROM {t}", c) if ada else None


def _tabel_hasil(conn: Any) -> list[str]:
    if isinstance(conn, KoneksiPG):
        sql = ("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema() "
               "AND table_name LIKE 'hasil\\_%' ORDER BY table_name")
        return [r[0] for r in conn.execute(sql).fetchall()]
    return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' "
                                       "AND name LIKE 'hasil\\_%' ESCAPE '\\' ORDER BY name")]


# Tabel data operasional & hasil proses yang dikosongkan saat sistem diatur ulang (uji dari nol). log_proses dan
# log_akses tidak dikosongkan (jejak audit, KNF-09); meta menyimpan versi skema.
TABEL_ATUR_ULANG = ("krs_asli", "ip_asli", "histori", "transaksi", "aturan", "support_item", "rating_bloom",
                    "batch_llm", "cl_mk", "rencana", "override_cpmk", "override_prasyarat", "ringkasan", "berkas")


def kosongkan_data(cfg: Config, pertahankan: Iterable[str] = ()) -> dict[str, int]:
    """Atur ulang basis data ke kondisi awal: kosongkan ``TABEL_ATUR_ULANG`` (kecuali ``pertahankan``) dan hapus
    tabel ``hasil_*``. Mengembalikan jumlah baris yang dihapus per tabel. Satu transaksi."""
    tetap = set(pertahankan)
    hasil: dict[str, int] = {}
    with koneksi(cfg) as c:
        for t in TABEL_ATUR_ULANG:
            if t in tetap:
                continue
            hasil[t] = int(c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            c.execute(f"DELETE FROM {t}")
        for t in _tabel_hasil(c):
            hasil[t] = int(c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            c.execute(f"DROP TABLE {t}")
    return hasil


# ---------------------------------------------------------------------
# Ekspor CSV (untuk JASP / lampiran)
# ---------------------------------------------------------------------


def ekspor_csv(cfg: Config, nama_berkas: str, df: pd.DataFrame) -> Path:
    cfg.ekspor_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.ekspor_dir / nama_berkas
    df.to_csv(path, index=False)
    return path


def sidik_konteks(cfg: Config) -> str:
    """Penanda versi data: berubah bila ada proses/impor (setiap ringkasan proses), rating, atau override berubah;
    tidak berubah ketika rencana atau log ditulis. Dipakai UI sebagai kunci cache (konteks planner & status data),
    sehingga dihitung dengan SATU kueri (satu perjalanan jaringan ke basis data cloud)."""
    sql = ("SELECT b FROM ("
           "SELECT 'r:' || nama || '@' || diperbarui AS b FROM ringkasan "
           "UNION ALL SELECT 'rating:' || COUNT(*) || ':' || COALESCE(MAX(timestamp), '') FROM rating_bloom "
           "UNION ALL SELECT 'ovr_cpmk:' || COUNT(*) || ':' || COALESCE(MAX(diperbarui), '') FROM override_cpmk "
           "UNION ALL SELECT 'ovr_pr:' || COUNT(*) || ':' || COALESCE(MAX(diperbarui), '') FROM override_prasyarat"
           ") t ORDER BY b")
    with koneksi(cfg) as c:
        return "|".join(r[0] for r in c.execute(sql).fetchall())


# ---------------------------------------------------------------------
# Backend PostgreSQL (hosting cloud)
# ---------------------------------------------------------------------
# Skema relasional identik dengan SQLite (DDL yang sama), hanya tipe REAL dipetakan ke DOUBLE PRECISION agar
# presisi support/confidence/lift/IP tidak turun ke float 4-byte. Modul lain tetap memakai placeholder ``?``.

_SKEMA_SIAP: set[tuple[str, str]] = set()      # (url, skema) yang sudah diperiksa pada proses ini
_KUNCI_SKEMA = threading.Lock()


class BarisPG(tuple):
    """Baris hasil kueri yang dapat diakses lewat indeks maupun nama kolom (setara ``sqlite3.Row``)."""

    _kolom: dict[str, int] = {}

    def __getitem__(self, k):            # type: ignore[override]
        if isinstance(k, str):
            return tuple.__getitem__(self, self._kolom[k])
        return tuple.__getitem__(self, k)

    def keys(self) -> list[str]:
        return list(self._kolom)


def _pabrik_baris(cursor):
    nama = [d.name for d in (cursor.description or [])]
    peta = {n: i for i, n in enumerate(nama)}
    kelas = type("Baris", (BarisPG,), {"_kolom": peta})
    return lambda nilai: kelas(nilai)


def _sql_pg(sql: str) -> str:
    """Placeholder ``?`` → ``%s`` (``%`` literal di-escape). SQL aplikasi tidak memuat ``?`` di dalam string."""
    return sql.replace("%", "%%").replace("?", "%s")


# Tabel log yang diurutkan menurut urutan penyisipan (``ORDER BY rowid`` di SQLite). PostgreSQL tidak memiliki
# rowid implisit, sehingga kolom identitas bernama ``rowid`` ditambahkan dan disembunyikan dari ``SELECT *``.
_TABEL_ROWID = ("log_proses", "log_akses")


def _ddl_pg(ddl: str) -> str:
    ddl = ddl.replace(" REAL", " DOUBLE PRECISION").replace(" BLOB", " BYTEA")
    for t in _TABEL_ROWID:
        ddl = ddl.replace(f"CREATE TABLE IF NOT EXISTS {t} (\n",
                          f"CREATE TABLE IF NOT EXISTS {t} (\n    rowid BIGINT GENERATED ALWAYS AS IDENTITY,\n", 1)
    return ddl


def _nama_skema(cfg: Config) -> str:
    skema = (cfg.paths.db_schema or "").strip()
    if skema and not skema.replace("_", "").isalnum():
        raise ValueError(f"nama skema tidak sah: {skema!r}")
    return skema


class KoneksiPG:
    """Pembungkus koneksi psycopg 3 dengan antarmuka yang dipakai aplikasi (``execute`` ber-placeholder ``?``)."""

    def __init__(self, conn, jalur: str | None = None):
        self.conn = conn
        self.jalur = jalur
        self._pk: dict[str, list[str]] = {}

    def atur_jalur(self) -> None:
        """``SET LOCAL search_path`` di awal setiap transaksi (bila ``jalur`` diisi). Di balik pooler mode transaksi
        (Neon/Supabase ``-pooler``) setiap transaksi dapat berjalan di koneksi server lain, dan ``SET`` biasa (tingkat
        sesi) tertinggal di koneksi server yang dipakai bersama klien lain. ``SET LOCAL`` berakhir saat
        commit/rollback."""
        if self.jalur is None:
            return
        self.conn.execute(f"SET LOCAL search_path TO {self.jalur}")

    def execute(self, sql: str, params: Iterable[Any] = ()):
        params = list(params)
        cur = self.conn.cursor(row_factory=_pabrik_baris)
        cur.execute(_sql_pg(sql) if params else sql, params or None)
        return cur

    def executemany(self, sql: str, seq_params: Iterable[Iterable[Any]]):
        cur = self.conn.cursor(row_factory=_pabrik_baris)
        cur.executemany(_sql_pg(sql), [list(p) for p in seq_params])
        return cur

    def kolom(self, tabel: str) -> list[str]:
        cur = self.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
                           "AND table_name = ? AND column_name <> 'rowid' ORDER BY ordinal_position", (tabel,))
        return [r[0] for r in cur.fetchall()]

    def ada_tabel(self, tabel: str) -> bool:
        return self.execute("SELECT to_regclass(?) IS NOT NULL", (tabel,)).fetchone()[0]

    def kunci_primer(self, tabel: str) -> list[str]:
        if tabel not in self._pk:
            cur = self.execute(
                "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid "
                "AND a.attnum = ANY(i.indkey) WHERE i.indrelid = to_regclass(?) AND i.indisprimary "
                "ORDER BY array_position(i.indkey, a.attnum)", (tabel,))
            self._pk[tabel] = [r[0] for r in cur.fetchall()]
        return self._pk[tabel]

    def sql_upsert(self, tabel: str, kolom_isi: list[str], kolom_tabel: list[str]) -> str:
        """Padanan ``INSERT OR REPLACE`` SQLite: baris berkunci sama diganti utuh (kolom yang tidak diisi
        kembali ke nilai bawaan)."""
        pk = self.kunci_primer(tabel)
        dasar = f"INSERT INTO {tabel} ({', '.join(kolom_isi)}) VALUES ({', '.join('?' * len(kolom_isi))})"
        if not pk:
            return dasar
        lain = [k for k in kolom_tabel if k not in pk]
        if not lain:
            return dasar + f" ON CONFLICT ({', '.join(pk)}) DO NOTHING"
        setel = ", ".join(f"{k} = EXCLUDED.{k}" if k in kolom_isi else f"{k} = DEFAULT" for k in lain)
        return dasar + f" ON CONFLICT ({', '.join(pk)}) DO UPDATE SET {setel}"

    def baca_df(self, sql: str, params: Iterable[Any] = ()) -> pd.DataFrame:
        cur = self.conn.cursor()
        params = list(params)
        cur.execute(_sql_pg(sql) if params else sql, params or None)
        kolom = [d.name for d in cur.description]
        df = pd.DataFrame.from_records(cur.fetchall(), columns=kolom, coerce_float=True)
        return df.drop(columns="rowid") if "rowid" in df.columns else df     # setara SQLite: rowid tidak ikut SELECT *

    def ganti_tabel_df(self, tabel: str, df: pd.DataFrame) -> None:
        """Padanan ``DataFrame.to_sql(if_exists="replace")`` tanpa SQLAlchemy (tipe mengikuti dtype pandas)."""
        def tipe(s: pd.Series) -> str:
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
                return "BIGINT"
            if pd.api.types.is_float_dtype(s):
                return "DOUBLE PRECISION"
            return "TEXT"
        kolom = [str(c) for c in df.columns]
        definisi = ", ".join(f'"{k.replace(chr(34), "")}" {tipe(df[c])}' for k, c in zip(kolom, df.columns))
        self.conn.execute(f"DROP TABLE IF EXISTS {tabel}")
        self.conn.execute(f"CREATE TABLE {tabel} ({definisi})")
        if df.empty:
            return
        nama = ", ".join(f'"{k.replace(chr(34), "")}"' for k in kolom)
        sql = f"INSERT INTO {tabel} ({nama}) VALUES ({', '.join(['%s'] * len(kolom))})"
        teks = [c for c in df.columns if tipe(df[c]) == "TEXT"]

        def sel(v, ke_teks):
            v = _nilai(v)
            return None if v is None else (str(v) if ke_teks and not isinstance(v, str) else v)
        baris = [[sel(v, c in teks) for c, v in zip(df.columns, r)] for r in df.itertuples(index=False, name=None)]
        with self.conn.cursor() as cur:
            cur.executemany(sql, baris)

    def commit(self) -> None:
        self.conn.commit()
        self.atur_jalur()              # transaksi berikutnya pada koneksi ini

    def rollback(self) -> None:
        self.conn.rollback()

    def close(self) -> None:
        self.conn.close()


# Kumpulan koneksi per URL (psycopg_pool): koneksi dipakai ulang antarpemanggilan ``koneksi()``. Membuka koneksi baru
# ke PostgreSQL cloud (TCP + TLS + autentikasi) ± 6× lebih lama daripada satu kueri, dan satu tampilan Streamlit
# memanggil ``koneksi()`` belasan kali. Koneksi yang diam > POOL_MAX_IDLE ditutup sebelum Neon menidurkan
# komputasinya (± 5 menit); koneksi yang diam > POOL_PERIKSA_SETELAH diuji dulu sebelum dipinjamkan.
POOL_MAX_IDLE = 240.0
POOL_PERIKSA_SETELAH = 60.0
_POOL: dict[str, Any] = {}
_KUNCI_POOL = threading.Lock()
_DIPAKAI: "weakref.WeakKeyDictionary[Any, float]" = weakref.WeakKeyDictionary()


def _periksa_koneksi(raw) -> None:
    """``check`` pool: koneksi yang lama diam diuji dengan kueri ringan; yang putus diganti otomatis oleh pool."""
    if time.monotonic() - _DIPAKAI.get(raw, 0.0) > POOL_PERIKSA_SETELAH:
        from psycopg_pool import ConnectionPool

        ConnectionPool.check_connection(raw)


def _pool(url: str):
    with _KUNCI_POOL:
        p = _POOL.get(url)
        if p is None:
            try:
                from psycopg_pool import ConnectionPool
            except ImportError as e:     # pragma: no cover - bergantung lingkungan
                raise RuntimeError("Backend PostgreSQL membutuhkan paket psycopg: "
                                   "pip install \"psycopg[binary,pool]\"") from e
            # prepare_threshold=None: tanpa prepared statement server-side, agar aman di balik PgBouncer mode
            # transaksi (connection string *-pooler* Neon/Supabase)
            p = ConnectionPool(url, min_size=0, max_size=8, open=True, name="crs-pti", check=_periksa_koneksi,
                               max_idle=POOL_MAX_IDLE, timeout=60,
                               kwargs={"connect_timeout": 30, "application_name": "crs-pti",
                                       "prepare_threshold": None})
            atexit.register(p.close)
            _POOL[url] = p
        return p


def _lewat_pooler(url: str) -> bool:
    from urllib.parse import urlsplit

    return "-pooler" in (urlsplit(url).hostname or "")


@contextmanager
def _koneksi_pg(cfg: Config) -> Iterator[KoneksiPG]:
    """Satu transaksi pada koneksi pinjaman pool: commit bila sukses, rollback bila gagal (oleh pool)."""
    url = cfg.paths.db_url.strip()
    skema = _nama_skema(cfg)
    # search_path per transaksi (SET LOCAL) hanya bila perlu: skema eksplisit, atau URL *-pooler* yang koneksi
    # servernya dipakai bersama klien lain. Koneksi langsung tanpa skema memakai bawaan server (tanpa round trip).
    jalur = skema or ('"$user", public' if _lewat_pooler(url) else None)
    with _pool(url).connection() as raw:
        conn = KoneksiPG(raw, jalur)
        kunci = (url, skema)
        if kunci not in _SKEMA_SIAP:
            with _KUNCI_SKEMA:
                if skema:
                    raw.execute(f"CREATE SCHEMA IF NOT EXISTS {skema}")
                conn.atur_jalur()
                _siapkan_pg(conn)
                conn.commit()
                _SKEMA_SIAP.add(kunci)
        else:
            conn.atur_jalur()
        yield conn
        _DIPAKAI[raw] = time.monotonic()


def _siapkan_pg(conn: KoneksiPG) -> None:
    """Buat/migrasikan skema. Kunci *advisory* mencegah dua proses (mis. dua sesi Streamlit) membuat tabel
    bersamaan."""
    conn.conn.execute("SELECT pg_advisory_xact_lock(hashtext('crs_skema'))")
    baru = not conn.ada_tabel("meta")
    if not baru:
        r = conn.execute("SELECT nilai FROM meta WHERE kunci = 'versi_skema'").fetchone()
        if r is not None and int(r[0]) == VERSI_SKEMA:
            return
    conn.conn.execute(_ddl_pg(DDL))
    r = conn.execute("SELECT nilai FROM meta WHERE kunci = 'versi_skema'").fetchone()
    if r is None:
        if not baru:
            raise RuntimeError("Basis data tanpa versi skema, bukan dibuat oleh aplikasi ini")
        conn.execute("INSERT INTO meta (kunci, nilai) VALUES (?, ?)", ("versi_skema", str(VERSI_SKEMA)))
        return
    versi = int(r[0])
    if versi > VERSI_SKEMA:
        raise RuntimeError(f"Basis data berskema versi {versi} lebih baru dari kode ({VERSI_SKEMA})")
    for v in range(versi + 1, VERSI_SKEMA + 1):
        for sql in MIGRASI.get(v, []):
            conn.conn.execute(sql.replace("ADD COLUMN ", "ADD COLUMN IF NOT EXISTS "))
        conn.execute("UPDATE meta SET nilai = ? WHERE kunci = 'versi_skema'", (str(v),))
    conn.conn.execute(_ddl_pg(DDL))


def id_skema_tes(path: str) -> str:
    """Nama skema PostgreSQL deterministik per lokasi basis data (isolasi tes)."""
    return "t_" + hashlib.md5(path.encode()).hexdigest()[:16]
