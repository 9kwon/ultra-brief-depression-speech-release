import argparse
import glob
import os
import re
from math import gcd

import librosa
import librosa.feature as lf
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt
from scipy.stats import iqr, linregress
from tqdm import tqdm

SR = 16000
WIN_MS, HOP_MS = 25, 10
FRAME = int(SR * WIN_MS / 1000)
HOP = int(SR * HOP_MS / 1000)
N_FFT = 512
HPF_CUTOFF = 30.0
MIN_DUR = 0.25
PYIN_WIDE = (70.0, 500.0)
VFRAC_MIN = 0.20
USE_FORMANT_WHEN_VFRAC = 0.60
EPS = 1e-9


def parse_filename(filename: str):
    """Parse filenames like: prt1001_d11_2_아기축구자전거.wav."""
    base = os.path.basename(filename)
    m = re.match(r'^(prt\d+)_d(\d+)_([0-9]+)_(.+)\.wav$', base)
    if not m:
        prt_match = re.match(r'^(prt\d+)', base)
        return {
            "prt": prt_match.group(1) if prt_match else "unknown",
            "day": np.nan,
            "seq": np.nan,
            "vocab": os.path.splitext(base)[0],
            "filename": base,
        }
    prt, day, seq, vocab = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    return {"prt": prt, "day": day, "seq": seq, "vocab": vocab, "filename": base}


def robust_stats(vec):
    v = np.asarray(vec, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    q5, q95 = np.percentile(v, [5, 95])
    return (float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v)), float(q95 - q5))


def spectral_flux_mag(y, sr, n_fft=512, hop_length=160, center=True):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=center))
    denom = np.maximum(S.sum(axis=0, keepdims=True), 1e-12)
    S = S / denom
    D = np.diff(S, axis=1)
    D[D < 0] = 0.0
    flux = np.sqrt((D ** 2).sum(axis=0))
    return np.concatenate([[0.0], flux])


def snr_est_from_rms(y):
    rms = lf.rms(y=y, frame_length=FRAME, hop_length=HOP).flatten()
    if rms.size < 5:
        return np.nan
    n = np.percentile(rms, 10)
    s = np.percentile(rms, 90)
    if n <= 0:
        return np.nan
    return 20 * np.log10((s + 1e-9) / (n + 1e-9))


def adapt_f0_bounds(y, sr, fmin_wide=70, fmax_wide=500):
    f0, _, vprob = librosa.pyin(y, sr=sr, fmin=fmin_wide, fmax=fmax_wide, frame_length=FRAME, hop_length=HOP)
    if f0 is None or np.all(np.isnan(f0)):
        return fmin_wide, fmax_wide, np.nan, 0.0, (f0, vprob)

    voiced = np.isfinite(f0)
    vf = float(np.mean(voiced))
    if vf < 0.15:
        return fmin_wide, fmax_wide, np.nan, vf, (f0, vprob)

    med = float(np.nanmedian(f0))
    fmin_ad = max(60.0, 0.55 * med)
    fmax_ad = min(600.0, 2.2 * med)
    if fmax_ad - fmin_ad < 80:
        pad = (80 - (fmax_ad - fmin_ad)) / 2
        fmin_ad = max(50.0, fmin_ad - pad)
        fmax_ad = min(600.0, fmax_ad + pad)
    return fmin_ad, fmax_ad, med, vf, (f0, vprob)


def hz_to_st_rel(x_hz, ref_hz):
    return 12.0 * np.log2((x_hz + EPS) / (ref_hz + EPS))


def summarize_pitch(pyin_f0, pyin_prob, ref_hz):
    if pyin_f0 is None:
        return dict(pitch_slope_st_per_s=np.nan, pitch_range_st=np.nan, pitch_conf_mean=np.nan)
    st = np.full_like(pyin_f0, np.nan, dtype=float)
    msk = np.isfinite(pyin_f0) & (pyin_f0 > 0) & np.isfinite(ref_hz) & (ref_hz > 0)
    st[msk] = hz_to_st_rel(pyin_f0[msk], ref_hz)
    t = np.arange(len(st)) * HOP / SR
    pitch_conf_mean = float(np.nanmean(pyin_prob)) if pyin_prob is not None else np.nan
    valid = np.isfinite(st)
    if valid.sum() >= 3:
        slope = float(linregress(t[valid], st[valid]).slope)
        prange = float(np.nanmax(st[valid]) - np.nanmin(st[valid]))
    else:
        slope, prange = np.nan, np.nan
    return dict(pitch_slope_st_per_s=slope, pitch_range_st=prange, pitch_conf_mean=pitch_conf_mean)


def _robust_hnr_from_harmonicity(harm_values_1d):
    v = np.asarray(harm_values_1d, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    v = v[v > -100.0]
    if v.size == 0:
        return np.nan
    return float(np.median(v))


def praat_basic_core(y, sr, fmin, fmax, voiced_frac_hint=np.nan, force_formant=False):
    snd = parselmouth.Sound(y, sr)
    pitch = call(snd, "To Pitch", 0.0, fmin, fmax)
    f0_pm = pitch.selected_array["frequency"]
    pm_vf = float(np.mean(f0_pm > 0)) if f0_pm.size else np.nan

    hnr = np.nan
    try:
        harm = call(snd, "To Harmonicity (cc)", 0.01, fmin, 0.1, 1.0)
        if harm.values.size:
            hnr = _robust_hnr_from_harmonicity(harm.values[0])
    except Exception:
        pass

    cpps = np.nan
    try:
        pcep = call(snd, "To PowerCepstrogram", fmin, 5000.0, 50.0)
        cpps = float(call(pcep, "Get CPPS", 0, 0, 60.0, 330.0, 0.1))
    except Exception:
        pass

    F1m = F1s = F2m = F2s = F3m = F3s = np.nan
    use_fm = voiced_frac_hint if np.isfinite(voiced_frac_hint) else pm_vf
    try:
        if force_formant or (use_fm is not None and use_fm >= USE_FORMANT_WHEN_VFRAC):
            fm = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50.0)
            times = np.arange(0, snd.get_total_duration(), 0.01)

            def getk(k):
                arr = []
                for t in times:
                    v = call(fm, "Get value at time", k, t, "Hertz", "Linear")
                    arr.append(np.nan if v == 0 else v)
                return np.asarray(arr, float)

            F1, F2, F3 = getk(1), getk(2), getk(3)
            F1m, F1s = float(np.nanmean(F1)), float(np.nanstd(F1))
            F2m, F2s = float(np.nanmean(F2)), float(np.nanstd(F2))
            F3m, F3s = float(np.nanmean(F3)), float(np.nanstd(F3))
    except Exception:
        pass

    return {
        "voiced_frac_praat": pm_vf,
        "hnr_mean_db": hnr,
        "cpps": cpps,
        "F1_mean": F1m, "F1_std": F1s,
        "F2_mean": F2m, "F2_std": F2s,
        "F3_mean": F3m, "F3_std": F3s,
    }


def librosa_feats_func(y, sr):
    rms = lf.rms(y=y, frame_length=FRAME, hop_length=HOP).flatten()
    zcr = lf.zero_crossing_rate(y, frame_length=FRAME, hop_length=HOP).flatten()
    sc = lf.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP).flatten()
    sb = lf.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP).flatten()
    ro = lf.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, roll_percent=0.95).flatten()
    fx = spectral_flux_mag(y, sr, n_fft=N_FFT, hop_length=HOP).flatten()

    rms_mean, rms_std, _, _, rms_iqr = robust_stats(rms)
    zcr_mean, zcr_std, _, _, zcr_iqr = robust_stats(zcr)
    sc_mean, sc_std, _, _, sc_iqr = robust_stats(sc)
    sb_mean, sb_std, _, _, sb_iqr = robust_stats(sb)
    ro_mean, ro_std, _, _, ro_iqr = robust_stats(ro)
    fx_mean, fx_std, _, _, fx_iqr = robust_stats(fx)

    mfcc = lf.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
    d = librosa.feature.delta(mfcc, order=1)
    dd = librosa.feature.delta(mfcc, order=2)

    out = {
        "snr_rms_est_db": snr_est_from_rms(y),
        "rms_mean": rms_mean, "rms_std": rms_std, "rms_iqr": rms_iqr,
        "zcr_mean": zcr_mean, "zcr_std": zcr_std, "zcr_iqr": zcr_iqr,
        "spec_centroid_mean": sc_mean, "spec_centroid_std": sc_std, "spec_centroid_iqr": sc_iqr,
        "spec_bandwidth_mean": sb_mean, "spec_bandwidth_std": sb_std, "spec_bandwidth_iqr": sb_iqr,
        "rolloff95_mean": ro_mean, "rolloff95_std": ro_std, "rolloff95_iqr": ro_iqr,
        "spec_flux_mean": fx_mean, "spec_flux_std": fx_std, "spec_flux_iqr": fx_iqr,
    }
    for i in range(13):
        out[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        out[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
        out[f"d_mfcc{i+1}_mean"] = float(np.mean(d[i]))
        out[f"d_mfcc{i+1}_std"] = float(np.std(d[i]))
        out[f"dd_mfcc{i+1}_mean"] = float(np.mean(dd[i]))
        out[f"dd_mfcc{i+1}_std"] = float(np.std(dd[i]))
    return out


def am_band_energies(y, sr, ds_fs=200, bands=((4, 8), (8, 16))):
    env = np.abs(hilbert(y)).astype(np.float64)
    g = gcd(int(sr), int(ds_fs))
    env_ds = resample_poly(env, ds_fs // g, sr // g)
    out = {}
    nyq = 0.5 * ds_fs
    for (lo, hi) in bands:
        lo_n = max(lo / nyq, 1e-6)
        hi_n = min(hi / nyq, 0.999999)
        if not (0 < lo_n < hi_n < 1):
            out[f"am_{lo}_{hi}_rms"] = np.nan
            out[f"am_{lo}_{hi}_std"] = np.nan
            continue
        sos = butter(4, [lo_n, hi_n], btype="bandpass", output="sos")
        z = sosfiltfilt(sos, env_ds)
        out[f"am_{lo}_{hi}_rms"] = float(np.sqrt(np.mean(z ** 2)))
        out[f"am_{lo}_{hi}_std"] = float(np.std(z))
    return out


def teager_energy_mean(y):
    if len(y) < 3:
        return np.nan
    y = y.astype(float)
    psi = y[1:-1] ** 2 - y[:-2] * y[2:]
    return float(np.mean(psi))


def sample_entropy(x, m=2, r_ratio=0.2):
    x = np.asarray(x, float)
    if x.size < 10:
        return np.nan
    r = r_ratio * np.std(x)

    def phi(mm):
        N = len(x) - mm + 1
        if N <= 1:
            return 0.0
        emb = np.stack([x[i:i + mm] for i in range(N)], axis=0)
        C = 0
        for i in range(N):
            d = np.max(np.abs(emb - emb[i]), axis=1)
            C += (d <= r).sum() - 1
        return C / (N * (N - 1))

    A, B = phi(m + 1), phi(m)
    return float(-np.log((A + 1e-12) / (B + 1e-12)))


def collect_wavs(target_root: str, subfolder: str):
    pattern = os.path.join(target_root, "sliced*", subfolder, "*.wav")
    wavs = glob.glob(pattern)
    if wavs:
        return sorted(wavs)
    pattern2 = os.path.join(target_root, "**", subfolder, "*.wav")
    return sorted(glob.glob(pattern2, recursive=True))


def extract_one_core(wav_path: str, mode: str = "diary"):
    meta = parse_filename(os.path.basename(wav_path))
    try:
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
        sos = butter(4, HPF_CUTOFF, btype="high", fs=SR, output="sos")
        y = sosfiltfilt(sos, y).astype(np.float32)
        duration = librosa.get_duration(y=y, sr=SR)
        if duration < MIN_DUR:
            return None
    except Exception:
        return None

    row = {**meta, "path": wav_path, "duration": float(duration), "source": mode}
    try:
        fmin_ad, fmax_ad, f0_med, vf, (f0_py, vprob) = adapt_f0_bounds(y, SR, *PYIN_WIDE)
        row.update({
            "fmin_adapt": float(fmin_ad),
            "fmax_adapt": float(fmax_ad),
            "f0_med_guess_py": float(f0_med) if np.isfinite(f0_med) else np.nan,
            "voiced_frac": float(vf),
        })
        if vf >= VFRAC_MIN and np.isfinite(f0_med):
            f0_valid = f0_py[np.isfinite(f0_py)]
            row["f0_mean"] = float(np.mean(f0_valid))
            row["f0_std"] = float(np.std(f0_valid))
            row["f0_min"] = float(np.min(f0_valid))
            row["f0_max"] = float(np.max(f0_valid))
            row["f0_iqr"] = float(iqr(f0_valid))
            row.update(summarize_pitch(f0_py, vprob, f0_med))
        else:
            for k in [
                "f0_mean", "f0_std", "f0_min", "f0_max", "f0_iqr",
                "pitch_slope_st_per_s", "pitch_range_st", "pitch_conf_mean",
            ]:
                row[k] = np.nan

        force_formant = mode == "start_recording"
        row.update(praat_basic_core(y, SR, fmin_ad, fmax_ad, voiced_frac_hint=vf, force_formant=force_formant))
        row.update(librosa_feats_func(y, SR))
        row.update(am_band_energies(y, SR))
        row["teo_mean"] = teager_energy_mean(y)
        row["sampen"] = sample_entropy(y[::4])
        return row
    except Exception as exc:
        print("Feat error:", os.path.basename(wav_path), exc)
        return None


def run_all(target_root: str, save_diary_csv: str, save_start_csv: str):
    diary_wavs = collect_wavs(target_root, "diary")
    start_wavs = collect_wavs(target_root, "start_recording")
    print(f"[collect] diary wavs: {len(diary_wavs)}")
    print(f"[collect] start_recording wavs: {len(start_wavs)}")

    diary_rows = []
    for p in tqdm(diary_wavs, desc="Extract diary"):
        r = extract_one_core(p, mode="diary")
        if r is not None:
            diary_rows.append(r)

    start_rows = []
    for p in tqdm(start_wavs, desc="Extract start_recording"):
        r = extract_one_core(p, mode="start_recording")
        if r is not None:
            start_rows.append(r)

    df_diary = pd.DataFrame(diary_rows)
    df_start = pd.DataFrame(start_rows)

    if save_diary_csv:
        os.makedirs(os.path.dirname(save_diary_csv), exist_ok=True)
        df_diary.to_csv(save_diary_csv, index=False, encoding="utf-8-sig")
        print(f"[save] diary -> {save_diary_csv}  (n={len(df_diary)})")
    if save_start_csv:
        os.makedirs(os.path.dirname(save_start_csv), exist_ok=True)
        df_start.to_csv(save_start_csv, index=False, encoding="utf-8-sig")
        print(f"[save] start_recording -> {save_start_csv}  (n={len(df_start)})")
    return df_diary, df_start


def parse_args():
    parser = argparse.ArgumentParser(description="Extract candidate acoustic features from sliced WAV files.")
    parser.add_argument("--target-folder", required=True, help="Root sound folder containing sliced*/diary and sliced*/start_recording")
    parser.add_argument("--save-diary-csv", required=True, help="Output path for diary candidate features CSV")
    parser.add_argument("--save-start-csv", required=True, help="Output path for start_recording candidate features CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args.target_folder, args.save_diary_csv, args.save_start_csv)
