#!/usr/bin/env python3
"""
Standalone batch TCA runner — equivalent to Step 6 of Batch_TCA_Pipeline.ipynb.
Run from the project root:  python run_batch_tca.py

MEMORY STRATEGY: Each target's full spike matrix is 3–8 GB. To stay within
available RAM, we process ONE target at a time: load its spike data once,
run both alignments sequentially (sharing the loaded data), then free
everything before loading the next target.
"""
import os, time, pickle, textwrap, gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import seaborn as sns

import scipy
if not hasattr(scipy, "random"):
    import numpy.random
    scipy.random = numpy.random
import tensortools as tt

# ── Paths ────────────────────────────────────────────────────────────
base = Path(__file__).resolve().parent
for p in [base, *base.parents]:
    if (p / "data").exists():
        project_root = p
        break
else:
    project_root = base

data_root = str(project_root / "data")
results_dir = project_root / "results" / "4replicas_10components_10frames"
results_dir.mkdir(parents=True, exist_ok=True)

# ── Targets and buffers ─────────────────────────────────────────────
sup_bef   = "VR2_2021_03_20_1"
sup_aft   = "VR2_2021_04_06_1"
unsup_bef = "TX105_2022_10_08_2"
unsup_aft = "TX105_2022_10_19_2"

BUFFER_TUNNEL = (2, 8)
BUFFER_SOUND  = (5, 5)

# ── Configuration ───────────────────────────────────────────────────
TCA_INPUT  = "zscored"         # "zscored" or "residuals"
RANK_RANGE = range(1, 11)      # ranks 1–10
REPLICATES = 4
CORRECT_BY = "simple"          # "simple" or "bytrial"

TARGETS = [
    (sup_bef,   "sup_bef"),
    (sup_aft,   "sup_aft"),
    (unsup_bef, "unsup_bef"),
    (unsup_aft, "unsup_aft"),
]
ALIGNMENTS = [
    ("Trial_start_time", BUFFER_TUNNEL),
    ("SoundTime",        BUFFER_SOUND),
]


# ── Worker helpers ──────────────────────────────────────────────────

def _build_tensor(trial_timestamps, buffer, spiking_data, spiking_timestamps):
    """Build 3D tensor (n_neurons, n_time, n_trials)."""
    n_trials = len(trial_timestamps)
    n_time = buffer[0] + buffer[1] + 1
    n_neurons = spiking_data.shape[1]
    tensor = np.zeros((n_neurons, n_time, n_trials), dtype=np.float32)
    for index, timestamp in enumerate(trial_timestamps):
        idx = np.searchsorted(spiking_timestamps, timestamp, side="right")
        tensor[:, :, index] = spiking_data[idx - buffer[0] : idx + buffer[1] + 1, :].T
    return tensor


def _build_speed_matrix(trial_timestamps, buffer, spiking_timestamps, run_speed):
    """Speed matrix (n_time, n_trials) aligned with tensor windows."""
    n_time = buffer[0] + buffer[1] + 1
    idx = np.searchsorted(spiking_timestamps, trial_timestamps, side="right")
    return run_speed[np.arange(n_time)[:, None] + (idx - buffer[0])].astype(np.float32)


def _speed_correct(tensor, speed_matrix, n_neurons, n_time, n_trials, correct_by="simple"):
    """Regress out running speed; return residual tensor."""
    n_obs = n_time * n_trials
    intercept = np.ones((n_obs, 1), dtype=np.float64)
    speed_col = speed_matrix.ravel().reshape(-1, 1)
    if correct_by == "simple":
        X = np.hstack([intercept, speed_col])
    elif correct_by == "bytrial":
        trial_idx = np.repeat(np.arange(n_trials), n_time)
        trial_dummies = (trial_idx[:, None] == np.arange(1, n_trials)).astype(np.float64)
        X = np.hstack([intercept, speed_col, trial_dummies])
    else:
        raise ValueError('correct_by must be "simple" or "bytrial"')
    Y = tensor.reshape(n_neurons, n_obs).T
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    return (Y - X @ B).T.reshape(n_neurons, n_time, n_trials).astype(np.float32)


def _zscore_tensor(tensor):
    """Per-neuron robust z-score (median/MAD)."""
    flat = tensor.reshape(tensor.shape[0], -1)
    med = np.median(flat, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(flat - med), axis=1, keepdims=True)
    mad = np.where(mad > 1e-12, mad, 1e-12)
    return ((flat - med) / mad).reshape(tensor.shape).astype(np.float32)


def _shift_nonneg(tensor):
    """Per-neuron minimum shift so all values >= 0."""
    flat = tensor.reshape(tensor.shape[0], -1)
    neuron_mins = flat.min(axis=1, keepdims=True)
    return (flat - neuron_mins).reshape(tensor.shape).astype(np.float32)


def _canonical_factors(U, V, W, n0, n1, n2):
    """Return (U, V, W) each shaped (mode_dim, rank)."""
    if U.shape[0] == n0 and V.shape[0] == n1 and W.shape[0] == n2:
        return U, V, W
    if U.shape[1] == n0 and V.shape[1] == n1 and W.shape[1] == n2:
        return U.T, V.T, W.T
    raise ValueError(f"Unexpected CP factor shapes {[a.shape for a in (U, V, W)]} for dims ({n0},{n1},{n2})")


def _load_neuron_to_area(root, recording_name):
    """Load trans file -> Series mapping neuron index -> area name."""
    filename = os.path.join(root, recording_name[:-1] + "trans.npz")
    with np.load(filename, allow_pickle=True) as retin:
        retin_df = pd.DataFrame({"iarea": retin["iarea"]})
    retin_df["neuron"] = retin_df.index.astype(int)
    area_map = pd.Series({8: "V1", 0: "mHV", 1: "mHV", 2: "mHV", 9: "mHV",
                          5: "lHV", 6: "lHV", 3: "aHV", 4: "aHV"})
    retin_df["area"] = retin_df["iarea"].map(area_map).fillna("Other")
    return retin_df.set_index("neuron")["area"]


def _get_beh_path(target_file):
    """Return behavior .npy filename for a target file."""
    mapping = {
        "VR2_2021_03_20_1": "Beh_sup_train1_before_learning.npy",
        "VR2_2021_04_06_1": "Beh_sup_train1_after_learning.npy",
        "TX105_2022_10_08_2": "Beh_unsup_train1_before_learning.npy",
        "TX105_2022_10_19_2": "Beh_unsup_train1_after_learning.npy",
    }
    if target_file not in mapping:
        raise ValueError(f"Unknown target_file: {target_file}")
    return mapping[target_file]


def _plot_and_save_components(data, results_dir):
    """One-row-per-rank component grid (comp 1 only). Saves PNG, no display."""
    rank_range_list = list(data.get("rank_range", range(1, data["best_rank"] + 1)))
    n_ranks = len(rank_range_list)
    neuron_order = data["neuron_order"]
    frame_order = data["frame_order"]
    trial_order = data["trial_order"]
    trial_to_stim = data["trial_to_stim"]
    if isinstance(trial_to_stim, dict):
        trial_to_stim = pd.Series(trial_to_stim)
    nta = data.get("neuron_to_area")
    if nta is None:
        nta = pd.Series("n/a", index=neuron_order)
    elif isinstance(nta, dict):
        nta = pd.Series(nta)

    fig_height = max(2.5, min(2.2 * n_ranks, 28))
    fig, axes = plt.subplots(n_ranks, 3, figsize=(12, fig_height), sharex=False)
    if n_ranks == 1:
        axes = np.array([axes])

    title = (f"{data['dataset_id']} | {data['beh_field']} | "
             f"buf ({data['buffer'][0]},{data['buffer'][1]}) | "
             f"ranks {min(rank_range_list)}-{max(rank_range_list)} | "
             f"{data['replicates']} reps | best rank: {data['best_rank']} | "
             f"input: {data.get('tca_input', 'n/a')}")
    fig.suptitle(textwrap.fill(title, width=65), fontsize=9)
    stim_pal = {"circle1": "#DFAE32", "leaf1": "#02968a"}

    for k, r in enumerate(rank_range_list):
        rep_idx = data["best_rep_idx"].get(r, 0)
        facs = data["ensemble"].factors(r)[rep_idx]
        aU, aV, aW = _canonical_factors(*facs, len(neuron_order),
                                        len(frame_order), len(trial_order))
        u1, v1, w1 = aU[:, 0], aV[:, 0], aW[:, 0]

        ax0 = axes[k, 0]
        ndf = pd.DataFrame({"neuron": neuron_order, "loading": u1})
        ndf["area"] = ndf["neuron"].map(nta)
        ndf = ndf.sort_values(["area", "neuron"]).reset_index(drop=True)
        ndf["neuron_idx"] = np.arange(len(ndf))
        sns.scatterplot(data=ndf, x="neuron_idx", y="loading", hue="area",
                        s=10, alpha=0.85, ax=ax0, legend=(k == 0), edgecolor="None")
        ax0.set_title(f"U (neurons) rank {r}")
        ax0.set_xlabel("neurons"); ax0.set_ylabel("")
        ax0.set_ylim(round(float(u1.min()), 1), round(float(u1.max()), 1))
        ax0.set_yticks(np.linspace(ax0.get_ylim()[0], ax0.get_ylim()[1], 5))
        sns.despine(ax=ax0, top=True, right=True)

        ax1 = axes[k, 1]
        tdf = pd.DataFrame({"frame": frame_order, "loading": v1})
        sns.lineplot(data=tdf, x="frame", y="loading", ax=ax1,
                     color=sns.color_palette("deep")[0], linewidth=2)
        ax1.set_title(f"V (time) rank {r}")
        ax1.set_xlabel("time (frames)"); ax1.set_ylabel("")
        ax1.set_ylim(round(float(v1.min()), 1), round(float(v1.max()), 1))
        ax1.set_yticks(np.linspace(ax1.get_ylim()[0], ax1.get_ylim()[1], 5))
        ax1.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
        al_label = ("tunnel entrance" if data.get("beh_field") == "Trial_start_time"
                     else "sound cue" if data.get("beh_field") == "SoundTime"
                     else data.get("beh_field", "alignment"))
        ax1.axvline(0, color="gray", linestyle="--", linewidth=1.5, label=al_label)
        ax1.legend(loc="best", fontsize=8)
        sns.despine(ax=ax1, top=True, right=True)

        ax2 = axes[k, 2]
        wdf = pd.DataFrame({"ft_trInd": trial_order, "loading": w1})
        wdf["TrialStim"] = wdf["ft_trInd"].map(trial_to_stim)
        sns.scatterplot(data=wdf, x="ft_trInd", y="loading", hue="TrialStim",
                        s=10, alpha=0.85, ax=ax2, legend=(k == 0),
                        palette=stim_pal, edgecolor="None")
        ax2.set_title(f"W (trials) rank {r}")
        ax2.set_xlabel("trials"); ax2.set_ylabel("")
        ax2.set_ylim(round(float(w1.min()), 1), round(float(w1.max()), 1))
        ax2.set_yticks(np.linspace(ax2.get_ylim()[0], ax2.get_ylim()[1], 5))
        t_min, t_max = int(trial_order.min()), int(trial_order.max())
        n_ticks = min(8, t_max - t_min + 1)
        ticks = np.unique(np.round(np.linspace(t_min, t_max, num=n_ticks)).astype(int))
        if len(ticks) == 0 or ticks[-1] != t_max:
            ticks = np.append(ticks, t_max)
        ax2.set_xticks(ticks)
        ax2.set_xlim(t_min - 0.5, t_max + 0.5)
        sns.despine(ax=ax2, top=True, right=True)

    if n_ranks >= 1:
        axes[0, 0].legend(title="area", fontsize=8, loc="upper right")
        axes[0, 2].legend(title="TrialStim", fontsize=8, loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = results_dir / (
        f"tensor_components_grid_{data['dataset_label']}_{data['dataset_id']}_"
        f"{data['beh_field']}_buffer{data['buffer'][0]}-{data['buffer'][1]}_"
        f"ncp_hals_{data.get('tca_input', 'na')}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


# ── Alignment worker (receives already-loaded target data) ──────────

def _run_alignment(target_file, dataset_label, alignment_name, buffer_used,
                   spk, beh, ft, rspd, nta, data_root,
                   results_dir, rank_range, replicates,
                   correct_by, tca_input):
    """Full TCA pipeline for one alignment, given already-loaded data."""
    t0 = time.perf_counter()

    # Build tensor
    ts  = beh[alignment_name]
    raw = _build_tensor(ts, buffer_used, spk, ft)
    N, T, K = raw.shape

    # Speed correction
    sm   = _build_speed_matrix(ts, buffer_used, ft, rspd)
    corr = _speed_correct(raw, sm, N, T, K, correct_by=correct_by)
    del raw, sm

    # Optional z-score
    inp = _zscore_tensor(corr) if tca_input == "zscored" else corr
    del corr

    # Shift nonneg
    tca_tensor = _shift_nonneg(inp)
    del inp

    # Fit TCA
    ens = tt.Ensemble(fit_method="ncp_hals")
    ens.fit(tca_tensor, ranks=rank_range, replicates=replicates)

    # Best replicate per rank + elbow
    orig_norm = np.linalg.norm(tca_tensor)
    mean_errors, best_rep_idx = [], {}
    for r in rank_range:
        reps = ens.factors(r)
        best_idx, best_err, errs = 0, None, []
        for i, facs in enumerate(reps):
            U, V, W = _canonical_factors(*facs, N, T, K)
            Xhat = np.einsum("ir,jr,kr->ijk", U, V, W)
            err = np.linalg.norm(tca_tensor - Xhat) / (orig_norm + 1e-12)
            errs.append(err)
            if best_err is None or err < best_err:
                best_err, best_idx = err, i
        mean_errors.append(float(np.mean(errs)))
        best_rep_idx[int(r)] = int(best_idx)

    del tca_tensor

    x = np.arange(len(mean_errors), dtype=float) + 1.0
    y = np.array(mean_errors, dtype=float)
    m = (y[-1] - y[0]) / (x[-1] - x[0] + 1e-12)
    b = y[0] - m * x[0]
    dist = np.abs(m * x + b - y) / (np.sqrt(m * m + 1.0) + 1e-12)
    best_rank = max(1, int(x[np.argmax(dist)]))

    # Save pickle
    spike_path    = os.path.join(data_root, target_file + "_SVD_dec.npy")
    beh_path      = os.path.join(data_root, _get_beh_path(target_file))
    frame_order   = np.arange(-buffer_used[0], buffer_used[1] + 1, dtype=int)
    trial_order   = np.asarray(beh["trInd"][:K], dtype=int)
    trial_to_stim = dict(zip(trial_order, beh["TrialStim"][:K]))

    payload = dict(
        ensemble=ens, best_rank=int(best_rank), rank_range=list(rank_range),
        replicates=int(replicates),
        best_rep_idx={int(k): int(v) for k, v in best_rep_idx.items()},
        mean_errors=mean_errors,
        dataset_label=dataset_label, dataset_id=target_file,
        spike_file=spike_path, beh_path=beh_path,
        neuron_order=np.arange(N), frame_order=frame_order,
        trial_order=trial_order, beh_field=alignment_name,
        buffer=tuple(buffer_used), trial_to_stim=trial_to_stim,
        neuron_to_area=nta, tca_input=tca_input, correct_by=correct_by,
    )
    fname = (f"ensemble_{dataset_label}_{target_file}_{alignment_name}_"
             f"buffer{buffer_used[0]}-{buffer_used[1]}_ncp_hals_"
             f"r{min(rank_range)}-{max(rank_range)}_"
             f"rep{replicates}_{tca_input}.pkl")
    pkl_path = results_dir / fname
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save plot
    plot_path = _plot_and_save_components(payload, results_dir)

    del payload, ens
    gc.collect()

    elapsed = time.perf_counter() - t0
    return dict(dataset_label=dataset_label, target_file=target_file,
                alignment=alignment_name, buffer=buffer_used,
                shape=(N, T, K), best_rank=best_rank,
                pkl=fname, plot=plot_path.name, elapsed_s=elapsed)


# ── Main: one target at a time, both alignments per target ──────────

if __name__ == "__main__":
    rank_range = list(RANK_RANGE)
    n_total = len(TARGETS) * len(ALIGNMENTS)

    print(f"Batch TCA — {n_total} jobs (sequential per target), "
          f"ranks {min(RANK_RANGE)}-{max(RANK_RANGE)}, {REPLICATES} reps, "
          f"input={TCA_INPUT}, correct_by={CORRECT_BY}")
    print(f"Results -> {results_dir}\n", flush=True)

    t_wall = time.perf_counter()
    results_summary = []

    for target_idx, (target_file, dataset_label) in enumerate(TARGETS, 1):

        # Load target data (one-time cost per target)
        t_load = time.perf_counter()
        spike_path = os.path.join(data_root, target_file + "_SVD_dec.npy")
        beh_path   = os.path.join(data_root, _get_beh_path(target_file))

        svd = np.load(spike_path, allow_pickle=True).item()
        spk = (svd["U"].T @ svd["V"]).T
        del svd; gc.collect()

        beh_all = np.load(beh_path, allow_pickle=True).item()
        beh = beh_all[target_file]
        del beh_all
        ft   = beh["ft"][: spk.shape[0] + 1]
        rspd = beh["ft_RunSpeed"][: len(ft)]
        nta  = _load_neuron_to_area(data_root, target_file)

        load_s = time.perf_counter() - t_load
        print(f"[{target_idx}/{len(TARGETS)}] Loaded {dataset_label} "
              f"({target_file}): {spk.shape[1]} neurons, "
              f"{spk.nbytes/1024**3:.1f} GB — {load_s:.1f}s", flush=True)

        # Run both alignments sharing the loaded spike data
        for alignment_name, buffer_used in ALIGNMENTS:
            print(f"  -> {alignment_name} buffer {buffer_used} ... ",
                  end="", flush=True)
            res = _run_alignment(
                target_file, dataset_label, alignment_name, buffer_used,
                spk, beh, ft, rspd, nta, data_root,
                results_dir, rank_range, REPLICATES, CORRECT_BY, TCA_INPUT,
            )
            results_summary.append(res)
            print(f"done  shape {res['shape']}  best_rank={res['best_rank']}  "
                  f"{res['elapsed_s']:.1f}s", flush=True)

        # Free this target's spike matrix before loading the next
        del spk, beh, ft, rspd, nta
        gc.collect()
        print(flush=True)

    t_total = time.perf_counter() - t_wall

    print(f"{'='*90}")
    print(f"Batch complete: {len(results_summary)}/{n_total} in {t_total:.1f}s")
    print(f"{'='*90}")
    hdr = (f"{'Label':12s} | {'Alignment':20s} | {'Shape':>20s} | "
           f"{'Best R':>6s} | {'Time':>7s} | Pickle")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results_summary,
                    key=lambda x: (x["dataset_label"], x["alignment"])):
        s = f"{r['shape'][0]}x{r['shape'][1]}x{r['shape'][2]}"
        print(f"{r['dataset_label']:12s} | {r['alignment']:20s} | {s:>20s} | "
              f"{r['best_rank']:>6d} | {r['elapsed_s']:>6.1f}s | {r['pkl']}")
