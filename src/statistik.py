"""Statistik kesepakatan skor beban kognitif (evaluasi beban kognitif LLM vs pakar).

Fungsi murni (tanpa I/O) yang dipakai ``src/export.py`` sebagai pratinjau dan cek silang hasil
JASP. Analisis inferensial resmi tetap dijalankan di JASP dari berkas ekspor.

- ICC dua arah, penilai sebagai efek acak (McGraw & Wong, 1996; Ten Hove, Jorgensen & van der Ark,
  2024): ICC(A,1) = kesepakatan absolut penilai tunggal (Pers. 2.7), ICC(C,1) = konsistensi
  (Pers. 2.8), beserta komponen varians (objek, penilai, interaksi+galat) dan selang kepercayaan
  95% berbasis distribusi F. Kategori interpretasi: Koo & Li (2016).
- MAE (Pers. 2.10), analisis Bland-Altman (Pers. 2.9), dan uji t berpasangan.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def _anova_dua_arah(x: np.ndarray) -> tuple[float, float, float]:
    """Kuadrat tengah (MSR objek, MSC penilai, MSE interaksi+galat) untuk matriks n objek × k penilai."""
    n, k = x.shape
    rerata = x.mean()
    ss_r = k * ((x.mean(axis=1) - rerata) ** 2).sum()
    ss_c = n * ((x.mean(axis=0) - rerata) ** 2).sum()
    ss_e = ((x - rerata) ** 2).sum() - ss_r - ss_c
    return ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))


def kategori_koo_li(nilai: float | None) -> str | None:
    """Koo & Li (2016): < 0,50 buruk; 0,50–0,75 sedang; 0,75–0,90 baik; > 0,90 sangat baik."""
    if nilai is None or not math.isfinite(nilai):
        return None
    if nilai < 0.5:
        return "buruk"
    if nilai < 0.75:
        return "sedang"
    if nilai <= 0.90:
        return "baik"
    return "sangat baik"


def icc(data: pd.DataFrame | np.ndarray, alpha: float = 0.05) -> dict:
    """ICC(A,1) dan ICC(C,1) dua arah acak untuk matriks lengkap n objek × k penilai (baris dengan nilai
    kosong dibuang). Mengembalikan estimasi, CI (1−alpha), komponen varians, dan kategori Koo & Li."""
    x = np.asarray(pd.DataFrame(data).dropna(), dtype=float)
    n, k = x.shape if x.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return {"n_objek": int(n), "n_penilai": int(k), "catatan": "butuh ≥ 2 objek dan ≥ 2 penilai"}
    msr, msc, mse = _anova_dua_arah(x)
    var_s, var_r, var_sr = (msr - mse) / k, (msc - mse) / n, mse
    hasil = {"n_objek": int(n), "n_penilai": int(k), "MSR": msr, "MSC": msc, "MSE": mse,
             "var_objek": var_s, "var_penilai": var_r, "var_interaksi_galat": var_sr}
    if msr + (k - 1) * mse == 0:      # semua skor identik → ICC tidak terdefinisi
        return {**hasil, "catatan": "varians nol, ICC tidak terdefinisi"}
    icc_c = (msr - mse) / (msr + (k - 1) * mse)
    penyebut_a = msr + (k - 1) * mse + k * (msc - mse) / n
    if abs(penyebut_a) < 1e-12:       # kombinasi varians degeneratif (mis. hanya beda rerata penilai)
        return {**hasil, "ICC_C1": icc_c, "catatan": "penyebut ICC(A,1) nol, ICC(A,1) tidak terdefinisi"}
    icc_a = (msr - mse) / penyebut_a
    q = 1 - alpha / 2
    df_r, df_e = n - 1, (n - 1) * (k - 1)
    # CI ICC(C,1) (McGraw & Wong, kasus 3)
    if mse > 0:
        f_obs = msr / mse
        f_l, f_u = f_obs / stats.f.ppf(q, df_r, df_e), f_obs * stats.f.ppf(q, df_e, df_r)
        ci_c = ((f_l - 1) / (f_l + k - 1), (f_u - 1) / (f_u + k - 1))
    else:
        ci_c = (float("nan"), float("nan"))
    # CI ICC(A,1) (McGraw & Wong, kasus 2A; Satterthwaite)
    rho = icc_a
    if mse > 0 and 0 < rho < 1:
        a = k * rho / (n * (1 - rho))
        b = 1 + k * rho * (n - 1) / (n * (1 - rho))
        v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / df_e)
        f1, f2 = stats.f.ppf(q, df_r, v), stats.f.ppf(q, v, df_r)
        c = k * msc + (k * n - k - n) * mse
        ci_a = (n * (msr - f1 * mse) / (f1 * c + n * msr), n * (f2 * msr - mse) / (c + n * f2 * msr))
    else:
        ci_a = (float("nan"), float("nan"))
    # ICC(A,k): kesepakatan absolut rerata k penilai (McGraw & Wong, kasus 2A, k penilai)
    icc_ak = (msr - mse) / (msr + (msc - mse) / n)
    if mse > 0 and 0 < rho < 1:
        ci_ak = (n * (msr - f1 * mse) / (f1 * (msc - mse) + n * msr),
                 n * (f2 * msr - mse) / ((msc - mse) + n * f2 * msr))
    else:
        ci_ak = (float("nan"), float("nan"))
    # ICC(1,1): satu arah (penilai acak berbeda per objek; Shrout & Fleiss kasus 1)
    msw = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    icc_11 = (msr - msw) / (msr + (k - 1) * msw) if msr + (k - 1) * msw else float("nan")
    if msw > 0:
        f_1 = msr / msw
        fl, fu = f_1 / stats.f.ppf(q, df_r, n * (k - 1)), f_1 * stats.f.ppf(q, n * (k - 1), df_r)
        ci_11 = ((fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1))
    else:
        ci_11 = (float("nan"), float("nan"))
    # dekomposisi varians (komponen negatif dipotong ke 0 hanya untuk proporsi)
    komp = np.clip([var_s, var_r, var_sr], 0, None)
    proporsi = komp / komp.sum() if komp.sum() > 0 else np.full(3, np.nan)
    # ANOVA pengukuran berulang: efek penilai (F = MSC / MSE)
    f_pen = msc / mse if mse > 0 else float("nan")
    p_pen = float(stats.f.sf(f_pen, k - 1, df_e)) if mse > 0 else float("nan")
    return {**hasil, "ICC_A1": icc_a, "ICC_A1_ci95": ci_a, "ICC_C1": icc_c, "ICC_C1_ci95": ci_c,
            "ICC_Ak": icc_ak, "ICC_Ak_ci95": ci_ak, "ICC_11": icc_11, "ICC_11_ci95": ci_11,
            "proporsi_var_objek": float(proporsi[0]), "proporsi_var_penilai": float(proporsi[1]),
            "proporsi_var_galat": float(proporsi[2]),
            "F_penilai": f_pen, "df_penilai": (k - 1, df_e), "p_penilai": p_pen,
            "kategori_A1": kategori_koo_li(icc_a),
            "kategori_A1_ci95": [kategori_koo_li(ci_a[0]), kategori_koo_li(ci_a[1])]}


def mae(x, y) -> float | None:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    return float((d["x"] - d["y"]).abs().mean()) if len(d) else None


def bland_altman(x, y) -> dict:
    """Selisih d = x − y: bias (rata-rata d), SD d, batas kesepakatan 95% = bias ± 1,96·SD, dan korelasi
    selisih dengan rata-rata (indikasi bias proporsional)."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 2:
        return {"n": int(len(d)), "catatan": "butuh ≥ 2 pasangan"}
    selisih, rata = d["x"] - d["y"], (d["x"] + d["y"]) / 2
    bias, sd = float(selisih.mean()), float(selisih.std(ddof=1))
    r = float(np.corrcoef(rata, selisih)[0, 1]) if selisih.std() > 0 and rata.std() > 0 else None
    return {"n": int(len(d)), "bias": bias, "sd_selisih": sd, "loa_bawah": bias - 1.96 * sd,
            "loa_atas": bias + 1.96 * sd, "r_selisih_vs_rata": r}


def uji_t_berpasangan(x, y) -> dict:
    """Paired t-test x − y (Pers. 2.13) + uji Shapiro-Wilk selisih (pemilihan uji di JASP)."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3:
        return {"n": int(len(d)), "catatan": "butuh ≥ 3 pasangan"}
    selisih = d["x"] - d["y"]
    if selisih.std(ddof=1) == 0:
        return {"n": int(len(d)), "rata_selisih": float(selisih.mean()), "catatan": "selisih konstan"}
    t = stats.ttest_rel(d["x"], d["y"])
    sw = stats.shapiro(selisih)
    return {"n": int(len(d)), "rata_selisih": float(selisih.mean()), "t": float(t.statistic),
            "df": int(len(d) - 1), "p": float(t.pvalue), "shapiro_W": float(sw.statistic),
            "shapiro_p": float(sw.pvalue), "cohen_dz": float(selisih.mean() / selisih.std(ddof=1))}


# ---------------------------------------------------------------------
# Kesepakatan data kategori/ordinal dan perbedaan sistematis antarpenilai
# ---------------------------------------------------------------------


def _matriks(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    return np.asarray(pd.DataFrame(data).dropna(), dtype=float)


def fleiss_kappa(data: pd.DataFrame | np.ndarray, kategori: list | None = None, alpha: float = 0.05) -> dict:
    """Fleiss' kappa (Fleiss, 1971) untuk n objek × k penilai (label kategori), beserta kappa per kategori
    dan CI asimtotik (galat baku Fleiss, Nee & Landis, 1979). Hanya objek dengan penilaian lengkap."""
    x = _matriks(data)
    n, k = x.shape if x.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return {"n_objek": int(n), "catatan": "butuh ≥ 2 objek dan ≥ 2 penilai"}
    kat = list(kategori) if kategori is not None else sorted(np.unique(x).tolist())
    hitung = np.stack([(x == c).sum(axis=1) for c in kat], axis=1).astype(float)   # n × q
    p_j = hitung.sum(axis=0) / (n * k)
    p_i = ((hitung ** 2).sum(axis=1) - k) / (k * (k - 1))
    p_bar, pe = p_i.mean(), float((p_j ** 2).sum())
    if pe >= 1:
        return {"n_objek": int(n), "catatan": "semua penilaian satu kategori, kappa tidak terdefinisi"}
    kappa = (p_bar - pe) / (1 - pe)
    q_j = 1 - p_j
    pq = p_j * q_j
    jumlah_pq = float(pq.sum())
    per_kat = {}
    for c, pj, pqj, kol in zip(kat, p_j, pq, hitung.T):
        if pqj > 0:
            per_kat[c] = float(1 - (kol * (k - kol)).sum() / (n * k * (k - 1) * pqj))
    se = (math.sqrt(2) / (jumlah_pq * math.sqrt(n * k * (k - 1)))
          * math.sqrt(jumlah_pq ** 2 - float((pq * (q_j - p_j)).sum()))) if jumlah_pq > 0 else float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    return {"n_objek": int(n), "n_penilai": int(k), "kappa": float(kappa), "se": se,
            "ci95": (float(kappa - z * se), float(kappa + z * se)), "per_kategori": per_kat}


def krippendorff_alpha(data: pd.DataFrame | np.ndarray, level: str = "ordinal",
                       n_bootstrap: int = 0, seed: int = 42) -> dict:
    """Krippendorff's alpha (Krippendorff, 2004) via matriks koinsidensi; ``level`` nominal | ordinal | interval.
    Unit = baris, penilai = kolom; nilai kosong diperbolehkan (unit dengan < 2 nilai diabaikan). CI opsional
    lewat bootstrap unit (persentil 2,5–97,5); hasil bootstrap berbeda tipis dari JASP karena acak."""
    x = np.asarray(pd.DataFrame(data), dtype=float)

    def hitung(m: np.ndarray) -> float:
        nilai = np.unique(m[~np.isnan(m)])
        if len(nilai) < 2:
            return float("nan")
        idx = {v: i for i, v in enumerate(nilai)}
        o = np.zeros((len(nilai), len(nilai)))
        for baris in m:
            v = baris[~np.isnan(baris)]
            mu = len(v)
            if mu < 2:
                continue
            for a in range(mu):
                for b in range(mu):
                    if a != b:
                        o[idx[v[a]], idx[v[b]]] += 1 / (mu - 1)
        n_c = o.sum(axis=1)
        n_tot = n_c.sum()
        if level == "nominal":
            d = 1 - np.eye(len(nilai))
        elif level == "interval":
            d = (nilai[:, None] - nilai[None, :]) ** 2
        else:   # ordinal: jarak dari frekuensi marginal kategori di antara g dan h
            d = np.zeros((len(nilai), len(nilai)))
            for g in range(len(nilai)):
                for h in range(len(nilai)):
                    lo, hi = min(g, h), max(g, h)
                    d[g, h] = (n_c[lo:hi + 1].sum() - (n_c[g] + n_c[h]) / 2) ** 2
        do = (o * d).sum() / n_tot
        de = (np.outer(n_c, n_c) * d).sum() / (n_tot * (n_tot - 1))
        return float(1 - do / de) if de > 0 else float("nan")

    hasil = {"n_unit": int((np.sum(~np.isnan(x), axis=1) >= 2).sum()), "level": level, "alpha": hitung(x)}
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        sampel = [hitung(x[rng.integers(0, len(x), len(x))]) for _ in range(n_bootstrap)]
        sampel = [s for s in sampel if math.isfinite(s)]
        if sampel:
            hasil["ci95_bootstrap"] = (float(np.percentile(sampel, 2.5)), float(np.percentile(sampel, 97.5)))
            hasil["n_bootstrap"] = n_bootstrap
    return hasil


def _holm(p: list[float]) -> list[float]:
    """Koreksi Holm (1979) — p terkoreksi monoton, dibatasi 1."""
    urut = np.argsort(p)
    m, hasil, maks = len(p), [0.0] * len(p), 0.0
    for r, i in enumerate(urut):
        maks = max(maks, min(1.0, (m - r) * p[i]))
        hasil[i] = maks
    return hasil


def friedman_conover(data: pd.DataFrame) -> dict:
    """Uji Friedman (Pers. 2.20; dengan koreksi ties) + Kendall's W (Pers. 2.21) dan uji lanjut Conover
    (Conover, 1999) untuk setiap pasangan penilai, p dikoreksi Holm. ``data`` = n objek × k penilai (kolom
    bernama). Positif ``selisih_rerata_peringkat`` = penilai kedua cenderung memberi level lebih tinggi."""
    d = pd.DataFrame(data).dropna()
    n, k = d.shape
    if n < 2 or k < 3:
        return {"n_objek": int(n), "catatan": "butuh ≥ 2 objek dan ≥ 3 penilai"}
    chi2, p = stats.friedmanchisquare(*[d[c] for c in d.columns])
    r = d.rank(axis=1)
    rj = r.sum(axis=0)
    a1 = float((r ** 2).to_numpy().sum())
    df_e = (n - 1) * (k - 1)
    var = 2 * (n * a1 - float((rj ** 2).sum())) / df_e
    pasangan = []
    kolom = list(d.columns)
    for i in range(k):
        for j in range(i + 1, k):
            beda = float(rj.iloc[j] - rj.iloc[i])
            t = abs(beda) / math.sqrt(var) if var > 0 else float("inf")
            pasangan.append({"penilai_1": kolom[i], "penilai_2": kolom[j], "selisih_jumlah_peringkat": beda,
                             "t": t, "p": float(2 * stats.t.sf(t, df_e)) if math.isfinite(t) else 0.0})
    for b, ph in zip(pasangan, _holm([b["p"] for b in pasangan])):
        b["p_holm"] = ph
    return {"n_objek": int(n), "n_penilai": int(k), "chi2": float(chi2), "df": k - 1, "p": float(p),
            "kendall_W": float(chi2 / (n * (k - 1))), "conover_holm": pasangan}
