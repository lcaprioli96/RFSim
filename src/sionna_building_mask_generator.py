from __future__ import annotations

import numpy as np
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import os
import json

from sionna.rt import PlanarRadioMap
from scipy.ndimage import distance_transform_edt
from pathlib import Path

def find_ground_plane(
    scene,
    vertical_axis: int = 1,
):
    """
    vertical_axis:
        0 -> X is height
        1 -> Y is height
        2 -> Z is height
    """
    horizontal_axes = [
        axis for axis in range(3)
        if axis != vertical_axis
    ]

    candidates = scene.objects.items()

    def horizontal_area(item):
        _, obj = item

        bbox = obj.mi_mesh.bbox()

        bbox_min = [
            float(bbox.min.x),
            float(bbox.min.y),
            float(bbox.min.z),
        ]

        bbox_max = [
            float(bbox.max.x),
            float(bbox.max.y),
            float(bbox.max.z),
        ]

        extents = [
            bbox_max[i] - bbox_min[i]
            for i in range(3)
        ]

        return (
            extents[horizontal_axes[0]]
            * extents[horizontal_axes[1]]
        )

    ground_name, ground_object = max(
        candidates,
        key=horizontal_area,
    )

    print(f"Detected ground: {ground_name}")

    return ground_name, ground_object

def scalar_to_float(value):
    array = np.asarray(value)

    if array.ndim == 0:
        return float(array)

    return float(array.reshape(-1)[0])

def mi_point3_to_numpy(point):
    return np.array(
        [
            scalar_to_float(point.x),
            scalar_to_float(point.y),
            scalar_to_float(point.z),
        ],
        dtype=np.float32,
    )

def create_building_mask_from_rays(
    scene,
    ground_name,
    vertical_axis,
    radio_map_config,
    margin=10.0
):
    """
    Generate a building mask using actual scene geometry.

    Parameters
    ----------
    scene:
        Sionna RT scene.

    ground_name:
        Name of the previously detected ground object.

    vertical_axis:
        0 -> X is height
        1 -> Y is height
        2 -> Z is height

    margin:
        Distance above and below the scene bounding box used for rays.

    Returns
    -------
    building_mask:
        True where the first ray intersection is a building.

    free_mask:
        True where the first ray intersection is the ground.

    invalid_mask:
        True where the ray does not hit anything.

    cell_centers:
        Radio-map cell centres in world coordinates.
    """
    if vertical_axis not in (0, 1, 2):
        raise ValueError("vertical_axis must be 0, 1, or 2")
    
    objects_list = scene.objects

    radio_map_grid = PlanarRadioMap(
        scene=scene,
        center=radio_map_config["center"],
        orientation=radio_map_config["orientation"],
        size=radio_map_config["size"],
        cell_size=radio_map_config["cell_size"],
    )

    if ground_name not in objects_list:
        raise KeyError(
            f"Ground object '{ground_name}' not found. "
            f"Available objects: {list(objects_list.keys())}"
        )

    ground_object = objects_list[ground_name]

    # Shape: [rows, columns, 3].
    cell_centers = radio_map_grid.cell_centers.numpy().astype(
        np.float32
    )

    rows, columns, _ = cell_centers.shape
    flat_centers = cell_centers.reshape(-1, 3)
    number_of_rays = len(flat_centers)

    # Full scene bounds.
    scene_bbox = scene.mi_scene.bbox()

    bbox_min = mi_point3_to_numpy(scene_bbox.min)
    bbox_max = mi_point3_to_numpy(scene_bbox.max)

    ray_start_height = (
        bbox_max[vertical_axis] + margin
    )

    ray_end_height = (
        bbox_min[vertical_axis] - margin
    )

    maximum_distance = (
        ray_start_height - ray_end_height
    )

    # Keep the two horizontal coordinates of each cell,
    # but move the ray origin above the entire scene.
    ray_origins = flat_centers.copy()

    ray_origins[:, vertical_axis] = (
        ray_start_height
    )

    ray_directions = np.zeros_like(
        ray_origins,
        dtype=np.float32,
    )

    # Rays point downward along the selected height axis.
    ray_directions[:, vertical_axis] = -1.0

    origins_mi = mi.Point3f(
        ray_origins[:, 0],
        ray_origins[:, 1],
        ray_origins[:, 2],
    )

    directions_mi = mi.Vector3f(
        ray_directions[:, 0],
        ray_directions[:, 1],
        ray_directions[:, 2],
    )

    rays = mi.Ray3f(
        origins_mi,
        directions_mi,
    )

    rays.maxt = mi.Float(maximum_distance)

    # Vectorized intersection against the complete scene.
    interactions = scene.mi_scene.ray_intersect(
        rays
    )

    valid_hits = interactions.is_valid()

    # Convert each hit Mitsuba shape pointer to the same
    # object ID representation used by Sionna.
    hit_object_ids = dr.reinterpret_array(
        mi.UInt32,
        mi.ShapePtr(interactions.shape),
    )

    ground_hits = (
        valid_hits
        & (
            hit_object_ids
            == ground_object.object_id
        )
    )

    building_hits = (
        valid_hits
        & (
            hit_object_ids
            != ground_object.object_id
        )
    )

    invalid_hits = ~valid_hits

    dr.eval(
        ground_hits,
        building_hits,
        invalid_hits,
    )

    free_mask = ground_hits.numpy().astype(bool)
    building_mask = building_hits.numpy().astype(bool)
    invalid_mask = invalid_hits.numpy().astype(bool)

    free_mask = free_mask.reshape(rows, columns)
    building_mask = building_mask.reshape(rows, columns)
    invalid_mask = invalid_mask.reshape(rows, columns)

    return (
        building_mask,
        free_mask,
        invalid_mask,
        cell_centers,
    )

def inspect_mask(building_mask, free_mask, invalid_mask):
    print("Mask shape:", building_mask.shape)

    print(
        "Building cells:",
        np.count_nonzero(building_mask),
    )

    print(
        "Free cells:",
        np.count_nonzero(free_mask),
    )

    print(
        "Invalid/outside cells:",
        np.count_nonzero(invalid_mask),
    )

def show_mask(building_mask):
    plt.figure(figsize=(8, 8))

    plt.imshow(
        building_mask,
        origin="lower",
        interpolation="nearest",
    )

    plt.xlabel("Radio-map column")
    plt.ylabel("Radio-map row")
    plt.title("Building mask from ray intersections")
    plt.colorbar(label="Building")
    plt.show()

def save_building_mask(
    scene_id,
    out_dir,
    ground_name,
    vertical_axis,
    building_mask,
    free_mask,
    invalid_mask,
    cell_centers,
    radio_map_config
):

    np.savez_compressed(
        out_dir / "scene_masks.npz",

        # Masks
        building_mask=np.asarray(
            building_mask,
            dtype=bool,
        ),
        free_mask=np.asarray(
            free_mask,
            dtype=bool,
        ),
        invalid_mask=np.asarray(
            invalid_mask,
            dtype=bool,
        ),

        # Exact 3D coordinate associated with each grid cell
        cell_centers=np.asarray(
            cell_centers,
            dtype=np.float32,
        )
    )

    metadata = {
        "scene_id": scene_id,
        "file": "scene_masks.npz",
        "ground_name": ground_name,
        "vertical_axis": vertical_axis,
        "center": np.asarray(
            radio_map_config["center"]
        ).tolist(),

        "orientation": np.asarray(
            radio_map_config["orientation"]
        ).tolist(),

        "size": np.asarray(
            radio_map_config["size"]
        ).tolist(),

        "cell_size": np.asarray(
            radio_map_config["cell_size"]
        ).tolist(),
    }

    with open(f"{out_dir}/mask_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Building mask saved to: {out_dir}")

def build_mask(scene_id, scene, root_dir, vertical_axis, radio_map_config):

    out_dir = root_dir / "output" / "mask"
    os.makedirs(out_dir, exist_ok=True)
    
    ground_name, ground_object = find_ground_plane(scene, vertical_axis)

    building_mask, free_mask, invalid_mask, mask_cell_centers = create_building_mask_from_rays(
        scene,
        ground_name,
        vertical_axis,
        radio_map_config,
        margin=10.0
    )

    inspect_mask(building_mask, free_mask, invalid_mask)

    show_mask(building_mask)

    save_building_mask(scene_id, out_dir, ground_name, vertical_axis, building_mask, free_mask, invalid_mask, mask_cell_centers, radio_map_config)
