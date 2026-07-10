import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sionna.rt import RadioMapSolver
radiomap_config = {
    "max_depth": 5,
    "cell_size": [3, 3],
    "center": [0, 1.5, 0],
    "size": [500, 500],
    "orientation": [0, 0, np.pi / 2],
    "samples_per_tx": 10**7
}

def linear_to_db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))

def watt_to_dbm(x):
    return 10 * np.log10(np.maximum(x, 1e-30)) + 30

def to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)

def compute_radiomap(scene, radiomap_config):
    rm_solver = RadioMapSolver()

    rm = rm_solver(
        scene=scene,
        max_depth=radiomap_config["max_depth"],
        cell_size=radiomap_config["cell_size"],
        center=radiomap_config["center"],
        size=radiomap_config["size"],
        orientation=radiomap_config["orientation"],
        samples_per_tx=radiomap_config["samples_per_tx"]
    )

    return rm

def extract_radiomap_arrays(rm):
    arrays = {
        "path_gain_linear": to_numpy(rm.path_gain),
        "rss_w": to_numpy(rm.rss),
        "sinr_linear": to_numpy(rm.sinr),
        "cell_centers": to_numpy(rm.cell_centers)
    }

    return arrays

def convert_radiomap_arrays(arrays):
    converted = {
        "path_gain_db": linear_to_db(arrays["path_gain_linear"]),
        "rss_dbm": watt_to_dbm(arrays["rss_w"]),
        "sinr_db": linear_to_db(arrays["sinr_linear"])
    }

    return converted

def save_radiomap_arrays(out_dir, arrays, converted):
    os.makedirs(out_dir, exist_ok=True)

    for name, value in arrays.items():
        np.save(f"{out_dir}/{name}.npy", value)

    for name, value in converted.items():
        np.save(f"{out_dir}/{name}.npy", value)

def get_tx_metadata(scene):
    tx_metadata = []

    for tx_idx, (tx_name, tx) in enumerate(scene.transmitters.items()):
        tx_metadata.append({
            "tx_idx": tx_idx,
            "tx_name": tx_name,
            "position": to_numpy(tx.position).tolist(),
            "orientation": to_numpy(tx.orientation).tolist()
            if hasattr(tx, "orientation") else None,
            "power_dbm": float(tx.power_dbm)
            if hasattr(tx, "power_dbm") else None
        })

    return tx_metadata

def save_radiomap_metadata(
    out_dir,
    scene_id,
    radiomap_config,
    arrays,
    converted,
    tx_metadata
):
    metadata = {
        "scene_id": scene_id,

        "radiomap_config": radiomap_config,

        "tx_metadata": tx_metadata,

        "array_shape": {
            name: list(value.shape)
            for name, value in {**arrays, **converted}.items()
        },

        "axis_meaning": {
            "axis_0": "tx_index",
            "axis_1": "radio_map_y_cell",
            "axis_2": "radio_map_x_cell"
        },

        "saved_arrays": list(arrays.keys()) + list(converted.keys())
    }

    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def build_radiomap_summary(scene_id, converted, radiomap_config):
    path_gain_db = converted["path_gain_db"]
    rss_dbm = converted["rss_dbm"]
    sinr_db = converted["sinr_db"]

    cell_size = radiomap_config["cell_size"]
    cell_area_m2 = cell_size[0] * cell_size[1]

    rows = []

    num_tx = rss_dbm.shape[0]

    for tx_idx in range(num_tx):
        rss_tx = rss_dbm[tx_idx]
        sinr_tx = sinr_db[tx_idx]
        path_gain_tx = path_gain_db[tx_idx]

        valid = np.isfinite(rss_tx)

        rows.append({
            "scene_id": scene_id,
            "tx_idx": tx_idx,

            "mean_rss_dbm": float(np.nanmean(rss_tx[valid])),
            "median_rss_dbm": float(np.nanmedian(rss_tx[valid])),
            "min_rss_dbm": float(np.nanmin(rss_tx[valid])),
            "max_rss_dbm": float(np.nanmax(rss_tx[valid])),

            "mean_sinr_db": float(np.nanmean(sinr_tx[valid])),
            "median_sinr_db": float(np.nanmedian(sinr_tx[valid])),

            "mean_path_gain_db": float(np.nanmean(path_gain_tx[valid])),

            "coverage_ratio_rss_gt_minus_80": float(np.mean(rss_tx[valid] > -80)),
            "coverage_ratio_rss_gt_minus_90": float(np.mean(rss_tx[valid] > -90)),
            "coverage_ratio_rss_gt_minus_100": float(np.mean(rss_tx[valid] > -100)),

            "covered_area_rss_gt_minus_90_m2": float(
                np.sum(rss_tx[valid] > -90) * cell_area_m2
            )
        })

    return pd.DataFrame(rows)

def save_radiomap_summary(out_dir, summary_df):
    summary_df.to_csv(f"{out_dir}/radio_map_summary.csv", index=False)

def plot_radiomap_array(array_2d, title, colorbar_label, out_path=None, vmin=None, vmax=None):
    plt.figure(figsize=(7, 6))
    plt.imshow(array_2d, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel("Cell x")
    plt.ylabel("Cell y")

    if out_path is not None:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.show()

def save_radiomap_plots(out_dir, converted):
    plots_dir = f"{out_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    path_gain_db = converted["path_gain_db"]
    rss_dbm = converted["rss_dbm"]
    sinr_db = converted["sinr_db"]

    num_tx = path_gain_db.shape[0]

    for tx_idx in range(num_tx):
        plot_radiomap_array(
            rss_dbm[tx_idx],
            title=f"RSS - TX {tx_idx}",
            colorbar_label="RSS [dBm]",
            out_path=f"{plots_dir}/rss_tx_{tx_idx}.png",
            vmin=-120,
            vmax=-40
        )

        plot_radiomap_array(
            sinr_db[tx_idx],
            title=f"SINR - TX {tx_idx}",
            colorbar_label="SINR [dB]",
            out_path=f"{plots_dir}/sinr_tx_{tx_idx}.png",
            vmin=-25,
            vmax=20
        )

    rss_max = np.max(rss_dbm, axis=0)
    sinr_max = np.max(sinr_db, axis=0)

    plot_radiomap_array(
        rss_max,
        title="Maximum RSS across TXs",
        colorbar_label="RSS [dBm]",
        out_path=f"{plots_dir}/rss_max.png",
        vmin=-120,
        vmax=-40
    )

    plot_radiomap_array(
        sinr_max,
        title="Maximum SINR across TXs",
        colorbar_label="SINR [dB]",
        out_path=f"{plots_dir}/sinr_max.png",
        vmin=-25,
        vmax=20
    )

def compute_scene_radiomap_dataset(
    scene_id,
    scene,
    out_dir,
    radiomap_config,
    save_plots=False
):
    os.makedirs(out_dir, exist_ok=True)

    print("Computing radio map...")
    rm = compute_radiomap(scene, radiomap_config)

    print("Extracting arrays...")
    arrays = extract_radiomap_arrays(rm)

    print("Converting arrays...")
    converted = convert_radiomap_arrays(arrays)

    print("Saving arrays...")
    save_radiomap_arrays(out_dir, arrays, converted)

    print("Saving metadata...")
    tx_metadata = get_tx_metadata(scene)
    save_radiomap_metadata(
        out_dir=out_dir,
        scene_id=scene_id,
        radiomap_config=radiomap_config,
        arrays=arrays,
        converted=converted,
        tx_metadata=tx_metadata
    )

    print("Building summary...")
    summary_df = build_radiomap_summary(
        scene_id=scene_id,
        converted=converted,
        radiomap_config=radiomap_config
    )
    save_radiomap_summary(out_dir, summary_df)

    if save_plots:
        print("Saving plots...")
        save_radiomap_plots(out_dir, converted)

    print("Done.")
    print("Saved files in:", out_dir)

    return {
        "rm": rm,
        "arrays": arrays,
        "converted": converted,
        "summary": summary_df,
        "tx_metadata": tx_metadata
    }

