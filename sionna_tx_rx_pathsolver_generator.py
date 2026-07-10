import numpy as np
from sionna.rt import PathSolver
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

paths_config = {
    "max_depth": 5,
    "samples_per_src": 10**6,
    "max_num_paths_per_src": 10**4,
    "synthetic_array": True,
    "los": True,
    "specular_reflection": True,
    "diffuse_reflection": True,
    "refraction": True,
    "diffraction": False,
    "normalize_delays": False
}

def linear_to_db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))

def dbm_to_watt(dbm):
    return 10 ** ((dbm - 30) / 10)

def watt_to_dbm(w):
    return 10 * np.log10(np.maximum(w, 1e-30)) + 30

def save_tx_rx_positions(scene_id, tx_list, rx_list, out_dir):
    tx_rows = []
    for tx in tx_list:
        tx_rows.append({
            "scene_id": scene_id,
            "tx_id": tx["id"],
            "tx_x": tx["position"][0],
            "tx_y": tx["position"][1],
            "tx_z": tx["position"][2],
            "tx_orientation_x": tx.get("orientation", [0,0,0])[0],
            "tx_orientation_y": tx.get("orientation", [0,0,0])[1],
            "tx_orientation_z": tx.get("orientation", [0,0,0])[2],
            "tx_power_dbm": tx.get("power_dbm", 30.0)
        })

    rx_rows = []
    for rx in rx_list:
        rx_rows.append({
            "scene_id": scene_id,
            "rx_id": rx["id"],
            "rx_x": rx["position"][0],
            "rx_y": rx["position"][1],
            "rx_z": rx["position"][2],
            "region_id": rx.get("region_id", None)
        })

    pd.DataFrame(tx_rows).to_csv(f"{out_dir}/tx_positions.csv", index=False)
    pd.DataFrame(rx_rows).to_csv(f"{out_dir}/rx_positions.csv", index=False)

def compute_paths(scene, paths_config):
    solver = PathSolver()

    paths = solver(
        scene=scene,
        max_depth=paths_config["max_depth"],
        samples_per_src=paths_config["samples_per_src"],
        max_num_paths_per_src=paths_config["max_num_paths_per_src"],
        synthetic_array=paths_config["synthetic_array"],
        los=paths_config["los"],
        specular_reflection=paths_config["specular_reflection"],
        diffuse_reflection=paths_config["diffuse_reflection"],
        refraction=paths_config["refraction"],
        diffraction=paths_config["diffraction"]
    )

    return paths

def extract_link_arrays(a, tau, rx_idx=0, tx_idx=0, rx_ant_idx=0, tx_ant_idx=0):
    a_link = a[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx]
    tau_link = tau[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx]

    a_link = np.squeeze(a_link)
    tau_link = np.squeeze(tau_link).reshape(-1)

    if a_link.ndim == 0:
        a_link = a_link.reshape(1)

    if a_link.ndim > 1:
        path_power_linear = np.mean(np.abs(a_link) ** 2, axis=-1)
    else:
        path_power_linear = np.abs(a_link) ** 2

    path_power_linear = path_power_linear.reshape(-1)

    n = min(len(path_power_linear), len(tau_link))
    path_power_linear = path_power_linear[:n]
    tau_link = tau_link[:n]

    valid = (
        np.isfinite(path_power_linear)
        & np.isfinite(tau_link)
        & (path_power_linear > 0)
    )

    return path_power_linear[valid], tau_link[valid]

def compute_native_link_metrics(path_power_linear, tau_link, tx_power_dbm):
    if len(path_power_linear) == 0:
        return {
            "num_paths": 0,
            "path_gain_linear": 0.0,
            "path_gain_db": -np.inf,
            "path_loss_db": np.inf,
            "rx_signal_power_dbm": -np.inf,
            "toa_s": np.nan,
            "max_delay_s": np.nan,
            "strongest_path_power_linear": np.nan,
            "strongest_path_power_db": np.nan,
            "strongest_path_delay_s": np.nan,
            "mean_delay_s": np.nan,
            "rms_delay_spread_s": np.nan
        }

    path_gain_linear = np.sum(path_power_linear)
    path_gain_db = linear_to_db(path_gain_linear)
    path_loss_db = -path_gain_db
    rx_signal_power_dbm = tx_power_dbm + path_gain_db

    strongest_idx = np.argmax(path_power_linear)
    strongest_path_power_linear = path_power_linear[strongest_idx]
    strongest_path_power_db = linear_to_db(strongest_path_power_linear)
    strongest_path_delay_s = tau_link[strongest_idx]

    toa_s = np.min(tau_link)
    max_delay_s = np.max(tau_link)

    weights = path_power_linear / path_gain_linear
    mean_delay_s = np.sum(weights * tau_link)
    rms_delay_spread_s = np.sqrt(
        np.sum(weights * (tau_link - mean_delay_s) ** 2)
    )

    return {
        "num_paths": int(len(path_power_linear)),
        "path_gain_linear": float(path_gain_linear),
        "path_gain_db": float(path_gain_db),
        "path_loss_db": float(path_loss_db),
        "rx_signal_power_dbm": float(rx_signal_power_dbm),
        "toa_s": float(toa_s),
        "max_delay_s": float(max_delay_s),
        "strongest_path_power_linear": float(strongest_path_power_linear),
        "strongest_path_power_db": float(strongest_path_power_db),
        "strongest_path_delay_s": float(strongest_path_delay_s),
        "mean_delay_s": float(mean_delay_s),
        "rms_delay_spread_s": float(rms_delay_spread_s)
    }

def compute_paper_rf_metrics(
    path_gain_linear,
    tx_power_dbm=30.0,
    n_re=1008,
    n_rb=84,
    thermal_noise_dbm=-100.0,
    interference_dbm=-110.0
):
    tx_power_w = dbm_to_watt(tx_power_dbm)
    signal_w = tx_power_w * path_gain_linear

    noise_w = dbm_to_watt(thermal_noise_dbm)
    interference_w = dbm_to_watt(interference_dbm)

    rssi_w = signal_w + interference_w + noise_w
    rssi_dbm = watt_to_dbm(rssi_w)

    nsinr_linear = signal_w / (interference_w + noise_w)
    nsinr_db = linear_to_db(nsinr_linear)

    nrsrp_w = signal_w / n_re
    nrsrp_dbm = watt_to_dbm(nrsrp_w)

    nrsrq_linear = (n_rb * nrsrp_w) / rssi_w
    nrsrq_db = linear_to_db(nrsrq_linear)

    return {
        "sim_rssi": float(rssi_dbm),
        "sim_nsinr": float(nsinr_db),
        "sim_nrsrp": float(nrsrp_dbm),
        "sim_nrsrq": float(nrsrq_db)
    }

def compute_path_rows(scene_id, tx_id, rx_id, path_power_linear, tau_link):
    path_rows = []

    if len(path_power_linear) == 0:
        return path_rows

    path_power_db = linear_to_db(path_power_linear)
    relative_delay_s = tau_link - np.min(tau_link)

    strongest_idx = np.argmax(path_power_linear)
    first_idx = np.argmin(tau_link)

    power_order = np.argsort(-path_power_linear)
    delay_order = np.argsort(tau_link)

    power_rank = np.empty_like(power_order)
    delay_rank = np.empty_like(delay_order)

    power_rank[power_order] = np.arange(len(power_order))
    delay_rank[delay_order] = np.arange(len(delay_order))

    for path_id in range(len(path_power_linear)):
        path_rows.append({
            "scene_id": scene_id,
            "tx_id": tx_id,
            "rx_id": rx_id,
            "path_id": int(path_id),
            "path_power_linear": float(path_power_linear[path_id]),
            "path_power_db": float(path_power_db[path_id]),
            "path_delay_s": float(tau_link[path_id]),
            "relative_delay_s": float(relative_delay_s[path_id]),
            "is_first_path": bool(path_id == first_idx),
            "is_strongest_path": bool(path_id == strongest_idx),
            "path_rank_by_power": int(power_rank[path_id]),
            "path_rank_by_delay": int(delay_rank[path_id])
        })

    return path_rows

def compute_scene_links(scene_id, scene, tx_list, rx_list, paths_config):
    all_link_rows = []
    all_path_rows = []

    for rx in rx_list:
        set_single_receiver(scene, rx)

        paths = compute_paths(scene, paths_config)

        a, tau = paths.cir(
            normalize_delays=paths_config["normalize_delays"],
            out_type="numpy"
        )

        for tx_idx, tx in enumerate(tx_list):
            path_power_linear, tau_link = extract_link_arrays(
                a=a,
                tau=tau,
                rx_idx=0,
                tx_idx=tx_idx
            )

            native_metrics = compute_native_link_metrics(
                path_power_linear=path_power_linear,
                tau_link=tau_link,
                tx_power_dbm=tx.get("power_dbm", 30.0)
            )

            paper_metrics = compute_paper_rf_metrics(
                path_gain_linear=native_metrics["path_gain_linear"],
                tx_power_dbm=tx.get("power_dbm", 30.0)
            )

            link_row = {
                "scene_id": scene_id,
                "tx_id": tx["id"],
                "rx_id": rx["id"],

                "tx_x": tx["position"][0],
                "tx_y": tx["position"][1],
                "tx_z": tx["position"][2],

                "rx_x": rx["position"][0],
                "rx_y": rx["position"][1],
                "rx_z": rx["position"][2],

                "tx_power_dbm": tx.get("power_dbm", 30.0)
            }

            link_row.update(native_metrics)
            link_row.update(paper_metrics)

            all_link_rows.append(link_row)

            path_rows = compute_path_rows(
                scene_id=scene_id,
                tx_id=tx["id"],
                rx_id=rx["id"],
                path_power_linear=path_power_linear,
                tau_link=tau_link
            )

            all_path_rows.extend(path_rows)

    return all_link_rows, all_path_rows

def save_scene_link_dataset(scene_id, out_dir, tx_list, rx_list, link_rows, path_rows, paths_config):
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(link_rows).to_csv(f"{out_dir}/link_data.csv", index=False)
    pd.DataFrame(path_rows).to_csv(f"{out_dir}/path_data.csv", index=False)

    save_tx_rx_positions(
        scene_id=scene_id,
        tx_list=tx_list,
        rx_list=rx_list,
        out_dir=out_dir
    )

    metadata = {
        "scene_id": scene_id,
        "num_tx": len(tx_list),
        "num_rx": len(rx_list),
        "num_links": len(link_rows),
        "num_paths_total": len(path_rows),
        "paths_config": paths_config
    }

    with open(f"{out_dir}/scene_link_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def build_fingerprint_table(link_rows):
    link_df = pd.DataFrame(link_rows)

    metrics = ["sim_rssi", "sim_nsinr", "sim_nrsrp", "sim_nrsrq"]

    base_cols = ["scene_id", "rx_id", "rx_x", "rx_y", "rx_z"]

    fingerprint = link_df[base_cols].drop_duplicates().copy()

    for metric in metrics:
        pivot = link_df.pivot(
            index="rx_id",
            columns="tx_id",
            values=metric
        )

        pivot.columns = [f"{metric}_{tx_id}" for tx_id in pivot.columns]
        pivot = pivot.reset_index()

        fingerprint = fingerprint.merge(pivot, on="rx_id", how="left")

    return fingerprint

def build_data(
        scene_id,
        scene,
        out_dir,
        paths_config
):
    
    tx_list = scene.transmitters
    rx_list = scene.receivers

    link_rows, path_rows = compute_scene_links(
        scene_id=scene_id,
        scene=scene,
        tx_list=tx_list,
        rx_list=rx_list,
        paths_config=paths_config
    )

    save_scene_link_dataset(
        scene_id=scene_id,
        out_dir=out_dir,
        tx_list=tx_list,
        rx_list=rx_list,
        link_rows=link_rows,
        path_rows=path_rows,
        paths_config=paths_config
    )

    fingerprint_df = build_fingerprint_table(link_rows)
    fingerprint_df.to_csv(f"{out_dir}/fingerprint_data.csv", index=False)

