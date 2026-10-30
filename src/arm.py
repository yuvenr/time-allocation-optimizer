"""Association rule mining dengan FP-Growth (UC-08; CLAUDE.md §6).

- Penambangan hanya memakai transaksi LATIH (keranjang per mahasiswa-semester, item ``kode@S``).
- Sumber per semester (``arm.sumber_data``): ``utamakan_asli`` (default) memakai KRS akademik nyata
  untuk semester yang transaksinya ≥ ``arm.min_transaksi_asli`` dan riwayat pelengkap untuk semester
  lain (mis. semester genap yang belum ada di ekspor KRS); ``asli`` / ``sintetis`` = satu sumber saja;
  ``gabungan`` = semua. Setiap aturan & support item mencatat ``sumber``-nya.
- Dengan ``arm.partisi_semester = true`` (default), transaksi dikelompokkan per semester dan
  support dihitung relatif terhadap jumlah transaksi semester itu (support global sebuah item
  paling tinggi ~1/8 karena setiap transaksi hanya memuat satu semester).
- Ambang EKSAK. mlxtend membandingkan support sebagai float (``ceil(min_support × n)``), sehingga
  itemset yang support-nya tepat di ambang dapat hilang (mis. 0,07 × 100 = 7,000000000000001 →
  8). Karena itu ambang diterjemahkan ke jumlah minimum ``k = ⌈min_support × n⌉`` dengan aritmetika
  pecahan, dan confidence/lift dihitung ulang dari jumlah (bilangan bulat) lalu disaring secara
  eksak. ``mlxtend`` tetap dipakai untuk FP-Growth dan pembangkitan kandidat aturan.
- Aturan yang konsekuennya memuat MK wajib (paket 08) dibuang (``arm.buang_konsekuen_wajib``)
  agar aturan trivial antar-MK wajib tidak mendominasi; ruang keputusan ARM = MK pilihan.
- Support item tunggal (``kode@S``) disimpan untuk SEMUA item (tanpa ambang) sebagai skor dasar planner.
- Setiap keluaran dicatat bersama sidik data masukan; ``ModelARM.dari_cfg`` mengabaikan hasil yang
  basi (data KRS/kurikulum berubah sejak penambangan).

Jalankan mandiri:  python -m src.arm [--sensitivitas]
"""
from __future__ import annotations

import inspect
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from src import db
from src.config import Config, load_config
from src.data_loader import Kurikulum, load_aturan, load_kurikulum, load_support_item, load_transaksi, sidik_raw
from src.logutil import catat_proses
from src.preprocessing import pisah_item
from src.schemas import Transaksi

SEP = ";"
KOLOM_ATURAN = ["anteseden", "konsekuen", "support", "confidence", "lift", "sumber", "run_id"]
_AR_PARAMS = inspect.signature(association_rules).parameters


def pilih_transaksi(transaksi: list[Transaksi], mode: str = "utamakan_asli",
                    min_asli: int = 20) -> tuple[list[Transaksi], dict[int, str]]:
    """Pilih transaksi per semester menurut ``mode``; mengembalikan (transaksi terpilih, semester → sumber)."""
    per: dict[int, dict[str, list[Transaksi]]] = defaultdict(lambda: defaultdict(list))
    for t in transaksi:
        per[t.semester_ke][t.sumber].append(t)
    hasil: list[Transaksi] = []
    sumber: dict[int, str] = {}
    for s in sorted(per):
        g = per[s]
        if mode == "gabungan":
            pilih = sorted(g)
        elif mode in ("asli", "sintetis"):
            pilih = [mode] if mode in g else []
        else:
            lain = sorted((x for x in g if x != "asli"), key=lambda x: -len(g[x]))
            pilih = ["asli"] if len(g.get("asli", ())) >= min_asli or not lain else lain[:1]
        if pilih:
            sumber[s] = "+".join(pilih)
            hasil += [t for x in pilih for t in g[x]]
    return hasil, sumber


def pilih_dari_cfg(transaksi: list[Transaksi], cfg: Config) -> tuple[list[Transaksi], dict[int, str]]:
    return pilih_transaksi(transaksi, cfg.arm.sumber_data, cfg.arm.min_transaksi_asli)


def _label_sumber(transaksi: list[Transaksi]) -> str:
    return "+".join(sorted({t.sumber for t in transaksi})) or "-"


def pecahan(x: float) -> Fraction:
    """Nilai ambang dari config (desimal) sebagai pecahan eksak: 0.07 → 7/100."""
    return Fraction(repr(float(x)))


def jumlah_minimum(min_support: float, n: int) -> int:
    """Jumlah transaksi minimum agar support ≥ min_support (eksak): ⌈min_support × n⌉, minimal 1."""
    return max(1, math.ceil(pecahan(min_support) * n))


def _kelompok(transaksi: list[Transaksi], partisi: bool) -> dict[str, list[Transaksi]]:
    if not partisi:
        return {"semua": list(transaksi)}
    grup: dict[str, list[Transaksi]] = defaultdict(list)
    for t in transaksi:
        grup[f"S{t.semester_ke}"].append(t)
    return dict(sorted(grup.items(), key=lambda kv: int(kv[0][1:])))


def _one_hot(item_lists: list[list[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    arr = te.fit(item_lists).transform(item_lists)
    return pd.DataFrame(arr, columns=te.columns_)


def _frequent(df: pd.DataFrame, min_support: float, max_len: int | None) -> pd.DataFrame:
    """Itemset dengan jumlah ≥ jumlah_minimum. Ambang yang dikirim ke mlxtend = (k − 0,5)/n agar
    ketiga perbandingan float di mlxtend setara dengan 'jumlah ≥ k' (jumlah selalu bilangan bulat)."""
    n = len(df)
    if n == 0 or df.shape[1] == 0 or not df.values.any():
        return pd.DataFrame(columns=["support", "itemsets"])
    k = jumlah_minimum(min_support, n)
    if k > n:
        return pd.DataFrame(columns=["support", "itemsets"])
    return fpgrowth(df, min_support=(k - 0.5) / n, use_colnames=True, max_len=max_len)


def _rules(fi: pd.DataFrame, n_transaksi: int, min_confidence: float, min_lift: float,
           lift_ketat: bool = False) -> pd.DataFrame:
    """Aturan dari itemset sering; confidence & lift dihitung ulang dari jumlah dan disaring eksak."""
    kosong = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    if fi.empty or (fi["itemsets"].map(len) > 1).sum() == 0:
        return kosong
    kw = {"metric": "confidence", "min_threshold": max(float(min_confidence) - 1e-9, 0.0)}
    if "num_itemsets" in _AR_PARAMS:
        kw["num_itemsets"] = n_transaksi
    # metrik 'certainty' internal mlxtend membagi nol bila support item = 1 (semester paket)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = association_rules(fi, **kw)
    if r.empty:
        return kosong
    n = n_transaksi
    c_ac = (r["support"] * n).round().astype(int)
    c_a = (r["antecedent support"] * n).round().astype(int)
    c_c = (r["consequent support"] * n).round().astype(int)
    conf_min, lift_min = pecahan(min_confidence), pecahan(min_lift)
    lolos = [Fraction(ac, a) >= conf_min
             and (Fraction(ac * n, a * c) > lift_min if lift_ketat else Fraction(ac * n, a * c) >= lift_min)
             for ac, a, c in zip(c_ac, c_a, c_c)]
    r = r.loc[lolos].copy()
    r["support"] = c_ac[lolos] / n
    r["confidence"] = c_ac[lolos] / c_a[lolos]
    r["lift"] = (c_ac[lolos] * n) / (c_a[lolos] * c_c[lolos])
    return r[["antecedents", "consequents", "support", "confidence", "lift"]]


def _konsekuen_wajib(konsekuen: str, kur: Kurikulum) -> bool:
    return any(pisah_item(i)[0] in kur.wajib for i in konsekuen.split(SEP))


def support_item(transaksi: list[Transaksi], partisi: bool = True) -> pd.DataFrame:
    """Support setiap item ``kode@S`` (popularitas MK pada semester tertentu), tanpa ambang."""
    baris = []
    for nama, grup in _kelompok(transaksi, partisi).items():
        n = len(grup)
        sumber = _label_sumber(grup)
        hitung = Counter(i for t in grup for i in set(t.items))
        for item, c in hitung.items():
            kode, s = pisah_item(item)
            baris.append({"item": item, "kode_mk": kode, "semester": s, "support": round(c / n, 6),
                          "jumlah": c, "n_transaksi": n, "sumber": sumber})
    df = pd.DataFrame(baris, columns=["item", "kode_mk", "semester", "support", "jumlah", "n_transaksi", "sumber"])
    return df.sort_values(["semester", "support", "item"], ascending=[True, False, True]).reset_index(drop=True)


def tambang(transaksi: list[Transaksi], min_support: float, min_confidence: float, min_lift: float = 1.0,
            max_len: int | None = 3, partisi: bool = True, run_id: str = "", kurikulum: Kurikulum | None = None,
            buang_konsekuen_wajib: bool = False, lift_ketat: bool = False) -> tuple[pd.DataFrame, dict]:
    """FP-Growth + association_rules (ambang eksak). ``lift_ketat`` = lift > min_lift (Gambar 2.4), selain itu
    lift ≥ min_lift. Mengembalikan (aturan, info per kelompok)."""
    semua = []
    info: dict[str, dict] = {}
    for nama, grup in _kelompok(transaksi, partisi).items():
        items = [t.items for t in grup]
        sumber = _label_sumber(grup)
        df = _one_hot(items)
        fi = _frequent(df, min_support, max_len)
        r = _rules(fi, len(items), min_confidence, min_lift, lift_ketat)
        info[nama] = {"sumber": sumber, "n_transaksi": len(items),
                      "jumlah_minimum": jumlah_minimum(min_support, len(items)),
                      "n_itemset": int(len(fi)), "n_aturan": int(len(r))}
        if not r.empty:
            semua.append(pd.DataFrame({
                "anteseden": r["antecedents"].map(lambda s: SEP.join(sorted(s))),
                "konsekuen": r["consequents"].map(lambda s: SEP.join(sorted(s))),
                "support": r["support"].round(6),
                "confidence": r["confidence"].round(6),
                "lift": r["lift"].round(6),
                "sumber": sumber,
                "run_id": run_id,
            }))
    aturan = pd.concat(semua, ignore_index=True) if semua else pd.DataFrame(columns=KOLOM_ATURAN)
    n_dibuang = 0
    if buang_konsekuen_wajib and kurikulum is not None and len(aturan):
        wajib = aturan["konsekuen"].map(lambda k: _konsekuen_wajib(k, kurikulum))
        n_dibuang = int(wajib.sum())
        aturan = aturan.loc[~wajib]
    info["_filter"] = {"buang_konsekuen_wajib": bool(buang_konsekuen_wajib and kurikulum is not None),
                       "n_aturan_konsekuen_wajib_dibuang": n_dibuang}
    # urutan total → keluaran deterministik
    aturan = aturan.sort_values(["lift", "confidence", "support", "anteseden", "konsekuen"],
                                ascending=[False, False, False, True, True]).reset_index(drop=True)
    return aturan[KOLOM_ATURAN], info


def sensitivity_analysis(transaksi: list[Transaksi], cfg: Config, kurikulum: Kurikulum | None = None) -> pd.DataFrame:
    """Grid min_support × min_confidence → jumlah aturan, rata-rata confidence & lift (UC-08).
    Pemilihan sumber & filter konsekuen wajib diterapkan sama seperti penambangan utama."""
    a = cfg.arm
    transaksi, _ = pilih_dari_cfg(transaksi, cfg)
    baris = []
    for sup in a.sensitivitas.support:
        for conf in a.sensitivitas.confidence:
            r, info = tambang(transaksi, sup, conf, a.min_lift, a.max_len, a.partisi_semester,
                              kurikulum=kurikulum, buang_konsekuen_wajib=a.buang_konsekuen_wajib,
                              lift_ketat=a.lift_ketat)
            baris.append({
                "min_support": sup,
                "min_confidence": conf,
                "n_itemset": int(sum(v["n_itemset"] for k, v in info.items() if not k.startswith("_"))),
                "n_aturan": int(len(r)),
                "rata_confidence": round(float(r["confidence"].mean()), 4) if len(r) else None,
                "rata_lift": round(float(r["lift"].mean()), 4) if len(r) else None,
            })
    return pd.DataFrame(baris)


def jalankan_arm(cfg: Config, min_support: float | None = None, min_confidence: float | None = None,
                 min_lift: float | None = None, max_len: int | None = None) -> dict:
    """UC-08: tambang aturan dari transaksi latih lalu simpan aturan & support item ke basis data."""
    a = cfg.arm
    param = {
        "min_support": min_support if min_support is not None else a.min_support,
        "min_confidence": min_confidence if min_confidence is not None else a.min_confidence,
        "min_lift": min_lift if min_lift is not None else a.min_lift,
        "lift_ketat": a.lift_ketat,
        "max_len": max_len if max_len is not None else a.max_len,
        "partisi_semester": a.partisi_semester,
        "buang_konsekuen_wajib": a.buang_konsekuen_wajib,
        "sumber_data": a.sumber_data,
        "min_transaksi_asli": a.min_transaksi_asli,
    }
    with catat_proses(cfg, "arm", param) as log:
        kur = load_kurikulum(cfg)
        latih, sumber_semester = pilih_dari_cfg(load_transaksi(cfg, "latih"), cfg)
        aturan, info = tambang(latih, param["min_support"], param["min_confidence"], param["min_lift"],
                               param["max_len"], param["partisi_semester"], run_id=log.run_id, kurikulum=kur,
                               buang_konsekuen_wajib=param["buang_konsekuen_wajib"], lift_ketat=param["lift_ketat"])
        sup = support_item(latih, param["partisi_semester"])
        db.hapus_ringkasan(cfg, "arm")   # hasil lama tidak sah selama penulisan
        db.ganti_isi(cfg, "aturan", aturan[KOLOM_ATURAN])
        db.ganti_isi(cfg, "support_item", sup)
        filt = info.pop("_filter")
        ringkasan = {
            "run_id": log.run_id,
            "parameter": param,
            "sidik_data": sidik_raw(cfg),
            "sumber_per_semester": {str(s): x for s, x in sumber_semester.items()},
            "n_transaksi_latih": len(latih),
            "n_aturan": int(len(aturan)),
            "n_aturan_per_sumber": {k: int(v) for k, v in aturan["sumber"].value_counts().items()},
            "n_aturan_konsekuen_wajib_dibuang": filt["n_aturan_konsekuen_wajib_dibuang"],
            "n_item": int(len(sup)),
            "rata_confidence": round(float(aturan["confidence"].mean()), 4) if len(aturan) else None,
            "rata_lift": round(float(aturan["lift"].mean()), 4) if len(aturan) else None,
            "per_kelompok": info,
        }
        log.ringkasan.update({k: v for k, v in ringkasan.items()
                              if k not in ("parameter", "run_id", "per_kelompok", "sumber_per_semester")})
        db.simpan_ringkasan(cfg, "arm", ringkasan)
    return ringkasan


def jalankan_sensitivitas(cfg: Config) -> pd.DataFrame:
    with catat_proses(cfg, "arm_sensitivitas", cfg.arm.sensitivitas.model_dump()) as log:
        tabel = sensitivity_analysis(load_transaksi(cfg, "latih"), cfg, load_kurikulum(cfg))
        db.simpan_hasil(cfg, "sensitivitas_arm", tabel)
        db.ekspor_csv(cfg, "arm_sensitivitas.csv", tabel)
        log.ringkasan["n_kombinasi"] = len(tabel)
    return tabel


# ---------------------------------------------------------------------
# Model untuk planner
# ---------------------------------------------------------------------


@dataclass
class ModelARM:
    """Support item tunggal + aturan berkonsekuen tunggal, diindeks per konsekuen; ``sumber_semester`` =
    asal transaksi tiap semester (asli | sintetis | …)."""

    support: dict[str, float] = field(default_factory=dict)
    per_konsekuen: dict[str, list[tuple[frozenset[str], float, float, str]]] = field(default_factory=dict)
    catatan: str = ""
    sumber_semester: dict[int, str] = field(default_factory=dict)

    @property
    def kosong(self) -> bool:
        return not self.support and not self.per_konsekuen

    @classmethod
    def dari_df(cls, aturan: pd.DataFrame | None, support: pd.DataFrame | None,
                kurikulum: Kurikulum | None = None) -> "ModelARM":
        """Bangun model; bila ``kurikulum`` diberikan, item berkode di luar kurikulum (mis. OLD:) diabaikan."""
        def dikenal(item: str) -> bool:
            return kurikulum is None or pisah_item(item)[0] in kurikulum

        m = cls()
        if support is not None and len(support):
            m.support = {i: float(s) for i, s in zip(support["item"], support["support"]) if dikenal(i)}
            if "sumber" in support.columns:
                m.sumber_semester = {int(s): str(x) for s, x in zip(support["semester"], support["sumber"])}
        if aturan is not None and len(aturan):
            for r in aturan.itertuples():
                kons = str(r.konsekuen).split(SEP)
                ant = frozenset(str(r.anteseden).split(SEP))
                if len(kons) != 1 or not all(dikenal(i) for i in ant | {kons[0]}):
                    continue
                teks = f"{{{', '.join(sorted(ant))}}} → {kons[0]} (conf {r.confidence:.2f}, lift {r.lift:.2f})"
                m.per_konsekuen.setdefault(kons[0], []).append((ant, float(r.confidence), float(r.lift), teks))
        return m

    @classmethod
    def dari_cfg(cls, cfg: Config, kurikulum: Kurikulum | None = None) -> "ModelARM":
        """Muat hasil ``jalankan_arm``; diabaikan (model kosong) bila belum ada atau basi."""
        ringkasan = db.baca_ringkasan(cfg, "arm")
        if ringkasan is None:
            return cls(catatan="belum ada hasil penambangan")
        if ringkasan.get("sidik_data") != sidik_raw(cfg):
            return cls(catatan="hasil penambangan basi (data KRS/kurikulum berubah), jalankan ulang ARM")
        return cls.dari_df(load_aturan(cfg), load_support_item(cfg), kurikulum)

    def skor_aturan(self, item: str, terpilih: set[str]) -> tuple[float, str | None]:
        """max(confidence × lift) atas aturan ``X → item`` dengan X ⊆ terpilih."""
        terbaik, teks = 0.0, None
        for ant, conf, lift, t in self.per_konsekuen.get(item, ()):
            if ant <= terpilih and conf * lift > terbaik:
                terbaik, teks = conf * lift, t
        return terbaik, teks


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = load_config()
    print(json.dumps(jalankan_arm(c), indent=2, ensure_ascii=False))
    if "--sensitivitas" in sys.argv:
        print(jalankan_sensitivitas(c).to_string(index=False))
