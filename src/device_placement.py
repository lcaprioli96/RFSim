import numpy as np
from sionna.rt import Transmitter, Receiver


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

def populate_scene_from_mask(
    scene,
    free_mask,
    invalid_mask,
    mask_cell_centers,
    num_tx,
    num_rx,
    vertical_axis=1,
    tx_height=6.0,
    rx_height=1.5,
    seed=None
):
    placement = sample_tx_rx_positions(
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

    add_devices_to_scene(
        scene=scene,
        tx_positions=placement["tx_positions"],
        rx_positions=placement["rx_positions"],
    )

    return placement