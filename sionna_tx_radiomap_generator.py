import numpy as np
from sionna.rt import RadioMapSolver
import matplotlib.pyplot as plt
import os
import json

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

def compute_radiomap(
        scene_id,
        scene,
        out_dir,
        radiomap_config,
        show=False,
):
    
    tx_names = list(scene.transmitters.keys())

    rm_solver = RadioMapSolver()

    rm = rm_solver(scene=scene,
        max_depth=radiomap_config["max_depth"],
        cell_size=radiomap_config["cell_size"],
        center=radiomap_config["center"],
        size=radiomap_config["size"],
        orientation=radiomap_config["orientation"],
        samples_per_tx=radiomap_config["samples_per_tx"]
    ) # rotate XY plane into XZ plane)

    path_gain = to_numpy(rm.path_gain)
    rss = to_numpy(rm.rss)
    sinr = to_numpy(rm.sinr)
    cell_centers = to_numpy(rm.cell_centers)

    print("path_gain:", path_gain.shape)
    print("rss:", rss.shape)
    print("sinr:", sinr.shape)
    print("cell_centers:", cell_centers.shape)

    if show:
        rm.show(metric="path_gain");
        rm.show(metric="rss");
        rm.show(metric="sinr");

    path_gain_db = linear_to_db(path_gain)
    rss_dbm = watt_to_dbm(rss)
    sinr_db = linear_to_db(sinr)

    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/path_gain_linear.npy", path_gain)
    np.save(f"{out_dir}/rss_w.npy", rss)
    np.save(f"{out_dir}/sinr_linear.npy", sinr)

    np.save(f"{out_dir}/path_gain_db.npy", path_gain_db)
    np.save(f"{out_dir}/rss_dbm.npy", rss_dbm)
    np.save(f"{out_dir}/sinr_db.npy", sinr_db)

    np.save(f"{out_dir}/cell_centers.npy", cell_centers)

    print("Saved files in:", out_dir)
    print(os.listdir(out_dir))

    metadata = {
        "scene_id": scene_id,
        "tx_names": tx_names,
        "metrics": ["path_gain_db", "rss_dbm", "sinr_db"],
        "array_shape": {
            "path_gain_db": list(path_gain_db.shape),
            "rss_dbm": list(rss_dbm.shape),
            "sinr_db": list(sinr_db.shape),
            "cell_centers": list(cell_centers.shape)
        },
        "axis_meaning": {
            "axis_0": "tx_index",
            "axis_1": "radio_map_y_cell",
            "axis_2": "radio_map_x_cell"
        },
        "radio_map_config": radiomap_config
    }

    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

