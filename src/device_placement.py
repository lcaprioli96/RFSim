import numpy as np
from sionna.rt import Transmitter, Receiver
import mitsuba as mi
import json
from pathlib import Path

def to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)

def point3_to_numpy(point):
    return np.array(
        [
            float(point.x),
            float(point.y),
            float(point.z),
        ],
        dtype=np.float64,
    )

def horizontal_distance(
    position_a,
    position_b,
    vertical_axis=1
):
    horizontal_axes = [
        axis for axis in range(3)
        if axis != vertical_axis
    ]

    difference = (
        position_a[horizontal_axes]
        - position_b[horizontal_axes]
    )

    return np.linalg.norm(difference)

def sample_mask_positions(
    free_mask,
    mask_cell_centers,
    num_positions,
    invalid_mask=None,
    seed=None,
    replace=False,
):
    """
    Sample world-coordinate positions from valid mask cells.

    Parameters
    ----------
    free_mask : np.ndarray
        Boolean mask with True for free cells.

    mask_cell_centers : np.ndarray
        Array of shape (H, W, 3) containing world coordinates.

    num_positions : int
        Number of positions to sample.

    invalid_mask : np.ndarray | None
        Optional mask with True for unusable cells.

    seed : int | None
        Random seed.

    replace : bool
        Whether the same cell can be sampled more than once.
        Usually False for TX/RX placement.
    """

    valid_mask = np.asarray(free_mask, dtype=bool).copy()

    if invalid_mask is not None:
        valid_mask &= ~np.asarray(invalid_mask, dtype=bool)

    valid_indices = np.argwhere(valid_mask)

    if not replace and num_positions > len(valid_indices):
        raise ValueError(
            f"Requested {num_positions} positions, but only "
            f"{len(valid_indices)} valid cells are available."
        )

    rng = np.random.default_rng(seed)

    selected = rng.choice(
        len(valid_indices),
        size=num_positions,
        replace=replace,
    )

    selected_indices = valid_indices[selected]

    positions = np.array([
        mask_cell_centers[row, col]
        for row, col in selected_indices
    ])

    return positions, selected_indices

def sample_tx_rx_positions(
    free_mask,
    mask_cell_centers,
    num_tx,
    num_rx,
    invalid_mask=None,
    tx_height=5.0,
    rx_height=1.5,
    vertical_axis=1,
    seed=None,
):
    total_devices = num_tx + num_rx

    positions, indices = sample_mask_positions(
        free_mask=free_mask,
        mask_cell_centers=mask_cell_centers,
        num_positions=total_devices,
        invalid_mask=invalid_mask,
        seed=seed,
        replace=False,
    )

    tx_positions = positions[:num_tx].copy()
    rx_positions = positions[num_tx:num_tx + num_rx].copy()

    tx_indices = indices[:num_tx]
    rx_indices = indices[num_tx:num_tx + num_rx]

    # Raise devices above the ground plane
    tx_positions[:, vertical_axis] += tx_height
    rx_positions[:, vertical_axis] += rx_height

    return {
        "tx_positions": tx_positions,
        "rx_positions": rx_positions,
        "tx_mask_indices": tx_indices,
        "rx_mask_indices": rx_indices,
    }

def add_devices_to_scene(
    scene,
    tx_positions,
    rx_positions,
    tx_orientation=(0.0, 0.0, 0.0),
    rx_orientation=(0.0, 0.0, 0.0),
):
    for tx_idx, position in enumerate(tx_positions):
        tx = Transmitter(
            name=f"tx_{tx_idx}",
            position=position.tolist(),
            orientation=list(tx_orientation),
        )
        scene.add(tx)

    for rx_idx, position in enumerate(rx_positions):
        rx = Receiver(
            name=f"rx_{rx_idx}",
            position=position.tolist(),
            orientation=list(rx_orientation),
        )
        scene.add(rx)

def sample_rooftop_tx_positions(
    scene,
    building_mask,
    mask_cell_centers,
    num_tx,
    vertical_axis=1,
    roof_clearance=1.0,
    min_building_height=2.0,
    min_tx_distance=10.0,
    roof_normal_threshold=0.5,
    ray_margin=10.0,
    seed=None,
):
    """
    Sample transmitter positions on building rooftops.

    Parameters
    ----------
    scene
        Sionna RT scene.

    building_mask : np.ndarray
        Boolean 2D array. True indicates a building cell.

    mask_cell_centers : np.ndarray
        Array with shape (H, W, 3), containing the world position
        corresponding to every mask cell.

    num_tx : int
        Number of transmitters to place.

    vertical_axis : int
        Scene vertical axis:
        0 -> X
        1 -> Y
        2 -> Z

    roof_clearance : float
        Distance added above the roof surface.

    min_building_height : float
        Minimum difference between the ground cell and the detected
        surface. This rejects rays that hit the ground instead of a roof.

    min_tx_distance : float
        Minimum horizontal distance between transmitters.

    roof_normal_threshold : float
        Minimum absolute vertical component of the surface normal.
        This helps reject steep walls or nearly vertical surfaces.

    ray_margin : float
        Distance above the scene bounding box from which rays start.

    seed : int | None
        Random seed.
    """

    building_mask = np.asarray(building_mask, dtype=bool)
    mask_cell_centers = np.asarray(mask_cell_centers)

    candidate_indices = np.argwhere(building_mask)

    if num_tx > len(candidate_indices):
        raise ValueError(
            f"Requested {num_tx} TX positions, but the building mask "
            f"contains only {len(candidate_indices)} building cells."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(candidate_indices)

    bbox = scene.mi_scene.bbox()
    bbox_max = point3_to_numpy(bbox.max)

    ray_direction = np.zeros(3, dtype=float)
    ray_direction[vertical_axis] = -1.0

    tx_positions = []
    tx_mask_indices = []
    roof_heights = []

    for row, col in candidate_indices:
        ground_position = np.asarray(
            mask_cell_centers[row, col],
            dtype=float,
        ).copy()

        ray_origin = ground_position.copy()
        ray_origin[vertical_axis] = (
            bbox_max[vertical_axis] + ray_margin
        )

        ray = mi.Ray3f(
            mi.Point3f(ray_origin),
            mi.Vector3f(ray_direction),
        )

        surface_interaction = scene.mi_scene.ray_intersect(ray)

        if not bool(surface_interaction.is_valid()):
            continue

        hit_position = np.asarray(
            to_numpy(surface_interaction.p),
            dtype=float,
        ).reshape(3)

        roof_height = (
            hit_position[vertical_axis]
            - ground_position[vertical_axis]
        )

        # Reject intersections with the ground plane
        if roof_height < min_building_height:
            continue

        surface_normal = np.asarray(
            to_numpy(surface_interaction.n),
            dtype=float,
        ).reshape(3)

        # Reject nearly vertical surfaces
        if abs(surface_normal[vertical_axis]) < roof_normal_threshold:
            continue

        tx_position = hit_position.copy()
        tx_position[vertical_axis] += roof_clearance

        # Enforce minimum horizontal spacing
        too_close = any(
            horizontal_distance(
                tx_position,
                existing_position,
                vertical_axis,
            ) < min_tx_distance
            for existing_position in tx_positions
        )

        if too_close:
            continue

        tx_positions.append(tx_position)
        tx_mask_indices.append([row, col])
        roof_heights.append(roof_height)

        if len(tx_positions) == num_tx:
            break

    if len(tx_positions) < num_tx:
        raise ValueError(
            f"Only {len(tx_positions)} valid rooftop positions were "
            f"found, but {num_tx} were requested. Try decreasing "
            f"min_tx_distance or min_building_height."
        )

    return {
        "roof_tx_positions": np.asarray(tx_positions),
        "roof_tx_mask_indices": np.asarray(tx_mask_indices),
        "roof_heights": np.asarray(roof_heights),
    }

def add_transmitters_to_scene(
    scene,
    tx_positions,
    orientation=(0.0, 0.0, 0.0),
    name_prefix="tx",
):
    for tx_idx, position in enumerate(tx_positions):
        transmitter = Transmitter(
            name=f"{name_prefix}_{tx_idx}",
            position=position.tolist(),
            orientation=list(orientation),
        )

        scene.add(transmitter)

def find_nearest_mask_cell(
    position,
    mask_cell_centers,
    vertical_axis=1,
):
    position = np.asarray(position, dtype=float)

    if position.shape != (3,):
        raise ValueError(
            f"Position must have shape (3,), got {position.shape}."
        )

    horizontal_axes = [
        axis for axis in range(3)
        if axis != vertical_axis
    ]

    target_horizontal = position[horizontal_axes]

    centers_horizontal = np.asarray(
        mask_cell_centers[..., horizontal_axes],
        dtype=float,
    )

    squared_distances = np.sum(
        (centers_horizontal - target_horizontal) ** 2,
        axis=-1,
    )

    row, col = np.unravel_index(
        np.argmin(squared_distances),
        squared_distances.shape,
    )

    distance = np.sqrt(squared_distances[row, col])

    return int(row), int(col), float(distance)

def get_rooftop_position_from_world_position(
    scene,
    world_position,
    ground_reference,
    vertical_axis=1,
    roof_clearance=1.0,
    min_building_height=2.0,
    roof_normal_threshold=0.5,
    ray_margin=10.0,
):
    world_position = np.asarray(
        world_position,
        dtype=float,
    ).copy()

    ground_reference = np.asarray(
        ground_reference,
        dtype=float,
    )

    bbox = scene.mi_scene.bbox()
    bbox_max = point3_to_numpy(bbox.max)

    # Preserve the exact horizontal Blender coordinates
    ray_origin = world_position.copy()
    ray_origin[vertical_axis] = (
        bbox_max[vertical_axis] + ray_margin
    )

    ray_direction = np.zeros(3, dtype=float)
    ray_direction[vertical_axis] = -1.0

    ray = mi.Ray3f(
        mi.Point3f(ray_origin.tolist()),
        mi.Vector3f(ray_direction.tolist()),
    )

    interaction = scene.mi_scene.ray_intersect(ray)

    if not bool(interaction.is_valid()):
        raise ValueError(
            f"No roof intersection found at position "
            f"{world_position.tolist()}."
        )

    hit_position = np.asarray(
        to_numpy(interaction.p),
        dtype=float,
    ).reshape(3)

    roof_height = (
        hit_position[vertical_axis]
        - ground_reference[vertical_axis]
    )

    if roof_height < min_building_height:

        print("requested_position:", world_position)
        print("ground_reference:", ground_reference)
        print("hit_position:", hit_position)
        print("vertical_axis:", vertical_axis)
        print(
            "roof height:",
            hit_position[vertical_axis] - ground_reference[vertical_axis],
        )
        print("Scene bbox min:", scene.mi_scene.bbox().min)
        print("Scene bbox max:", scene.mi_scene.bbox().max)

        raise ValueError(
            f"Detected surface is only {roof_height:.2f} m "
            "above the reference ground."
        )

    surface_normal = np.asarray(
        to_numpy(interaction.n),
        dtype=float,
    ).reshape(3)

    if abs(surface_normal[vertical_axis]) < roof_normal_threshold:
        raise ValueError(
            "The detected surface is probably a wall rather than a roof."
        )

    final_position = hit_position.copy()
    final_position[vertical_axis] += roof_clearance

    return final_position, roof_height

def resolve_position_from_masks(
    scene,
    requested_position,
    free_mask,
    building_mask,
    invalid_mask,
    mask_cell_centers,
    device_type,
    vertical_axis=1,
    ground_height=1.5,
    roof_clearance=1.0,
    min_building_height=2.0,
    max_cell_distance=None,
    allow_rooftop_rx=False,
):
    requested_position = np.asarray(
        requested_position,
        dtype=float,
    ).copy()

    row, col, cell_distance = find_nearest_mask_cell(
        position=requested_position,
        mask_cell_centers=mask_cell_centers,
        vertical_axis=vertical_axis,
    )

    if (
        max_cell_distance is not None
        and cell_distance > max_cell_distance
    ):
        raise ValueError(
            f"{device_type.upper()} position is {cell_distance:.2f} m "
            "from the nearest mask-cell center."
        )

    if invalid_mask is not None and invalid_mask[row, col]:
        raise ValueError(
            f"{device_type.upper()} position maps to invalid "
            f"cell ({row}, {col})."
        )

    if building_mask[row, col]:
        if device_type == "rx" and not allow_rooftop_rx:
            raise ValueError(
                f"RX position maps to building position {requested_position} in cell ({row}, {col})."
            )

        surface = "roof"

        final_position, roof_height = (
            get_rooftop_position_from_world_position(
                scene=scene,
                world_position=requested_position,
                ground_reference=mask_cell_centers[row, col],
                vertical_axis=vertical_axis,
                roof_clearance=roof_clearance,
                min_building_height=min_building_height,
            )
        )

    elif free_mask[row, col]:
        surface = "ground"
        roof_height = None

        # Preserve the exact Blender position
        final_position = requested_position.copy()

        # Optionally override only the vertical coordinate
        if ground_height is not None:
            final_position[vertical_axis] = (
                mask_cell_centers[row, col, vertical_axis]
                + ground_height
            )

    else:
        raise ValueError(
            f"{device_type.upper()} position maps to cell "
            f"({row}, {col}), which is neither free nor building."
        )

    metadata = {
        "device_type": device_type,
        "requested_position": requested_position.tolist(),
        "final_position": final_position.tolist(),
        "mask_index": [int(row), int(col)],
        "mask_cell_distance": float(cell_distance),
        "surface": surface,
        "roof_height": (
            float(roof_height)
            if roof_height is not None
            else None
        ),
    }

    return final_position, metadata

def build_random_placement_metadata(
    scene_id,
    ground_tx_positions,
    rooftop_tx_positions,
    rx_positions,
    ground_tx_indices,
    rooftop_tx_indices,
    rx_indices,
    roof_heights,
    vertical_axis,
    tx_height,
    rx_height,
    roof_clearance,
    min_building_height,
    min_tx_distance,
    seed,
    mask_filename="scene_masks.npz",
):
    tx_metadata = []
    rx_metadata = []

    # Ground transmitters
    for tx_idx, (position, mask_index) in enumerate(
        zip(ground_tx_positions, ground_tx_indices)
    ):
        tx_metadata.append({
            "tx_idx": int(tx_idx),
            "tx_name": f"tx_{tx_idx}",
            "placement_type": "ground",
            "position": np.asarray(position).tolist(),
            "mask_index": np.asarray(mask_index).tolist(),
            "height_above_ground": float(tx_height),
            "roof_height": None,
        })

    # Rooftop transmitters continue after ground TXs
    rooftop_start_idx = len(ground_tx_positions)

    for rooftop_idx, (position, mask_index, roof_height) in enumerate(
        zip(
            rooftop_tx_positions,
            rooftop_tx_indices,
            roof_heights,
        )
    ):
        tx_idx = rooftop_start_idx + rooftop_idx

        tx_metadata.append({
            "tx_idx": int(tx_idx),
            "tx_name": f"tx_{tx_idx}",
            "placement_type": "rooftop",
            "position": np.asarray(position).tolist(),
            "mask_index": np.asarray(mask_index).tolist(),
            "height_above_ground": None,
            "roof_height": float(roof_height),
            "roof_clearance": float(roof_clearance),
        })

    # Ground receivers
    for rx_idx, (position, mask_index) in enumerate(
        zip(rx_positions, rx_indices)
    ):
        rx_metadata.append({
            "rx_idx": int(rx_idx),
            "rx_name": f"rx_{rx_idx}",
            "placement_type": "ground",
            "position": np.asarray(position).tolist(),
            "mask_index": np.asarray(mask_index).tolist(),
            "height_above_ground": float(rx_height),
        })

    metadata = {
        "scene_id": str(scene_id),

        "placement_method": "random_mask_sampling",

        "mask_source": {
            "filename": str(mask_filename),
            "building_mask_key": "building_mask",
            "free_mask_key": "free_mask",
            "invalid_mask_key": "invalid_mask",
            "cell_centers_key": "mask_cell_centers",
        },

        "coordinate_system": {
            "position_order": ["x", "y", "z"],
            "vertical_axis": int(vertical_axis),
            "mask_index_order": ["row", "column"],
        },

        "placement_config": {
            "seed": None if seed is None else int(seed),
            "tx_height": float(tx_height),
            "rx_height": float(rx_height),
            "roof_clearance": float(roof_clearance),
            "min_building_height": float(min_building_height),
            "min_tx_distance": float(min_tx_distance),
            "sampling_with_replacement": False,
        },

        "device_ordering": {
            "transmitters": "ground transmitters first, rooftop transmitters second",
            "receivers": "receiver list order",
        },

        "summary": {
            "num_tx": int(
                len(ground_tx_positions)
                + len(rooftop_tx_positions)
            ),
            "num_ground_tx": int(len(ground_tx_positions)),
            "num_rooftop_tx": int(len(rooftop_tx_positions)),
            "num_rx": int(len(rx_positions)),
        },

        "transmitters": tx_metadata,
        "receivers": rx_metadata,
    }

    return metadata

def build_list_placement_metadata(
    scene_id,
    tx_metadata,
    rx_metadata,
    vertical_axis,
    ground_tx_height,
    ground_rx_height,
    roof_clearance,
    min_building_height,
    max_cell_distance,
    allow_rooftop_rx=False,
    mask_filename="scene_masks.npz",
):
    transmitters = []
    receivers = []

    for tx_idx, device in enumerate(tx_metadata):
        transmitters.append({
            "tx_idx": int(tx_idx),
            "tx_name": f"tx_{tx_idx}",
            "placement_type": device["surface"],

            # Original precise position from Blender
            "requested_position": device["requested_position"],

            # Position actually assigned in Sionna
            "position": device["final_position"],

            "mask_index": device["mask_index"],
            "mask_cell_distance": float(
                device["mask_cell_distance"]
            ),

            "height_above_ground": (
                None
                if ground_tx_height is None
                else float(ground_tx_height)
            ),

            "roof_height": (
                None
                if device["roof_height"] is None
                else float(device["roof_height"])
            ),

            "roof_clearance": (
                float(roof_clearance)
                if device["surface"] == "roof"
                else None
            ),
        })

    for rx_idx, device in enumerate(rx_metadata):
        receivers.append({
            "rx_idx": int(rx_idx),
            "rx_name": f"rx_{rx_idx}",
            "placement_type": device["surface"],

            "requested_position": device["requested_position"],
            "position": device["final_position"],

            "mask_index": device["mask_index"],
            "mask_cell_distance": float(
                device["mask_cell_distance"]
            ),

            "height_above_ground": (
                None
                if ground_rx_height is None
                else float(ground_rx_height)
            ),

            "roof_height": (
                None
                if device["roof_height"] is None
                else float(device["roof_height"])
            ),

            "roof_clearance": (
                float(roof_clearance)
                if device["surface"] == "roof"
                else None
            ),
        })

    num_ground_tx = sum(
        device["placement_type"] == "ground"
        for device in transmitters
    )

    num_rooftop_tx = sum(
        device["placement_type"] == "roof"
        for device in transmitters
    )

    num_ground_rx = sum(
        device["placement_type"] == "ground"
        for device in receivers
    )

    num_rooftop_rx = sum(
        device["placement_type"] == "roof"
        for device in receivers
    )

    return {
        "scene_id": str(scene_id),

        "placement_method": "position_list_with_mask_validation",

        "mask_source": {
            "filename": str(mask_filename),
            "building_mask_key": "building_mask",
            "free_mask_key": "free_mask",
            "invalid_mask_key": "invalid_mask",
            "cell_centers_key": "mask_cell_centers",
        },

        "coordinate_system": {
            "position_order": ["x", "y", "z"],
            "vertical_axis": int(vertical_axis),
            "mask_index_order": ["row", "column"],
        },

        "placement_config": {
            "ground_tx_height": (
                None
                if ground_tx_height is None
                else float(ground_tx_height)
            ),
            "ground_rx_height": (
                None
                if ground_rx_height is None
                else float(ground_rx_height)
            ),
            "roof_clearance": float(roof_clearance),
            "min_building_height": float(min_building_height),
            "max_cell_distance": (
                None
                if max_cell_distance is None
                else float(max_cell_distance)
            ),
            "allow_rooftop_rx": bool(allow_rooftop_rx),
        },

        "device_ordering": {
            "transmitters": "input transmitter list order",
            "receivers": "input receiver list order",
        },

        "summary": {
            "num_tx": int(len(transmitters)),
            "num_ground_tx": int(num_ground_tx),
            "num_rooftop_tx": int(num_rooftop_tx),
            "num_rx": int(len(receivers)),
            "num_ground_rx": int(num_ground_rx),
            "num_rooftop_rx": int(num_rooftop_rx),
        },

        "transmitters": transmitters,
        "receivers": receivers,
    }

def save_placement_metadata(
    out_dir,
    metadata,
    random=False
):
    if random:
        output_path = Path(out_dir) / "placement_metadata_random.json"
    else:
        output_path = Path(out_dir) / "placement_metadata_list.json"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

def populate_scene_random(
    scene_id,
    scene,
    root_dir,
    free_mask,
    building_mask,
    invalid_mask,
    mask_cell_centers,
    num_tx,
    num_rx,
    rooftops_num_tx,
    vertical_axis=1,
    tx_height=6.0,
    rx_height=1.5,
    roof_clearance=1.0,
    min_building_height=2.0,
    min_tx_distance=10.0,
    seed=None
):
    # Empty defaults ensure that concatenation always works
    ground_tx_positions = np.empty((0, 3), dtype=float)
    rx_positions = np.empty((0, 3), dtype=float)
    ground_tx_indices = np.empty((0, 2), dtype=int)
    rx_indices = np.empty((0, 2), dtype=int)

    rooftop_tx_positions = np.empty((0, 3), dtype=float)
    rooftop_tx_indices = np.empty((0, 2), dtype=int)
    roof_heights = np.empty((0,), dtype=float)

    if (num_tx + num_rx) > 0:
        ground_placement = sample_tx_rx_positions(
            free_mask=free_mask,
            invalid_mask=invalid_mask,
            mask_cell_centers=mask_cell_centers,
            num_tx=num_tx,
            num_rx=num_rx,
            tx_height=tx_height,
            rx_height=rx_height,
            vertical_axis=vertical_axis,
            seed=seed,
        )
        ground_tx_positions = ground_placement["tx_positions"]
        rx_positions = ground_placement["rx_positions"]

        ground_tx_indices = ground_placement["tx_mask_indices"]
        rx_indices = ground_placement["rx_mask_indices"]

    if rooftops_num_tx > 0:
        rooftop_placement = sample_rooftop_tx_positions(
            scene,
            building_mask,
            mask_cell_centers,
            rooftops_num_tx,
            vertical_axis=vertical_axis,
            roof_clearance=roof_clearance,
            min_building_height=min_building_height,
            min_tx_distance=min_tx_distance,
            seed=seed
        )
        rooftop_tx_positions = rooftop_placement["tx_positions"]
        rooftop_tx_indices = rooftop_placement["tx_mask_indices"]
        roof_heights = rooftop_placement["roof_heights"]

    # Ground TXs first, rooftop TXs second
    all_tx_positions = np.concatenate(
        [ground_tx_positions, rooftop_tx_positions],
        axis=0,
    )

    # Add everything once so TX names remain unique
    add_devices_to_scene(
        scene=scene,
        tx_positions=all_tx_positions,
        rx_positions=rx_positions,
    )

    metadata = build_random_placement_metadata(
        scene_id=scene_id,
        ground_tx_positions=ground_tx_positions,
        rooftop_tx_positions=rooftop_tx_positions,
        rx_positions=rx_positions,
        ground_tx_indices=ground_tx_indices,
        rooftop_tx_indices=rooftop_tx_indices,
        rx_indices=rx_indices,
        roof_heights=roof_heights,
        vertical_axis=vertical_axis,
        tx_height=tx_height,
        rx_height=rx_height,
        roof_clearance=roof_clearance,
        min_building_height=min_building_height,
        min_tx_distance=min_tx_distance,
        seed=seed,
    )

    save_placement_metadata(root_dir, metadata, True)

def populate_scene_from_list(
    scene_id,
    scene,
    root_dir,
    tx_positions,
    rx_positions,
    free_mask,
    building_mask,
    invalid_mask,
    mask_cell_centers,
    vertical_axis=1,
    ground_tx_height=6.0,
    ground_rx_height=1.5,
    roof_clearance=1.0,
    min_building_height=2.0,
    max_cell_distance=None,
    allow_rooftop_rx=False,
):
    resolved_tx_positions = []
    resolved_rx_positions = []

    tx_metadata = []
    rx_metadata = []

    for tx_idx, requested_position in enumerate(tx_positions):
        final_position, metadata = resolve_position_from_masks(
            scene=scene,
            requested_position=requested_position,
            free_mask=free_mask,
            building_mask=building_mask,
            invalid_mask=invalid_mask,
            mask_cell_centers=mask_cell_centers,
            device_type="tx",
            vertical_axis=vertical_axis,
            ground_height=ground_tx_height,
            roof_clearance=roof_clearance,
            min_building_height=min_building_height,
            max_cell_distance=max_cell_distance,
        )

        metadata["device_index"] = tx_idx

        resolved_tx_positions.append(final_position)
        tx_metadata.append(metadata)

    for rx_idx, requested_position in enumerate(rx_positions):
        final_position, metadata = resolve_position_from_masks(
            scene=scene,
            requested_position=requested_position,
            free_mask=free_mask,
            building_mask=building_mask,
            invalid_mask=invalid_mask,
            mask_cell_centers=mask_cell_centers,
            device_type="rx",
            vertical_axis=vertical_axis,
            ground_height=ground_rx_height,
            roof_clearance=roof_clearance,
            min_building_height=min_building_height,
            max_cell_distance=max_cell_distance,
            allow_rooftop_rx=allow_rooftop_rx,
        )

        metadata["device_index"] = rx_idx

        resolved_rx_positions.append(final_position)
        rx_metadata.append(metadata)

    resolved_tx_positions = np.asarray(
        resolved_tx_positions,
        dtype=float,
    ).reshape(-1, 3)

    resolved_rx_positions = np.asarray(
        resolved_rx_positions,
        dtype=float,
    ).reshape(-1, 3)

    add_devices_to_scene(
        scene=scene,
        tx_positions=resolved_tx_positions,
        rx_positions=resolved_rx_positions,
    )

    metadata = build_list_placement_metadata(
        scene_id=scene_id,
        tx_metadata=tx_metadata,
        rx_metadata=rx_metadata,
        vertical_axis=vertical_axis,
        ground_tx_height=ground_tx_height,
        ground_rx_height=ground_rx_height,
        roof_clearance=roof_clearance,
        min_building_height=min_building_height,
        max_cell_distance=max_cell_distance,
        allow_rooftop_rx=allow_rooftop_rx,
    )

    save_placement_metadata(root_dir, metadata, False)