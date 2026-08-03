import numpy as np


def point3_to_numpy(point):
    return np.array(
        [
            float(point.x),
            float(point.y),
            float(point.z),
        ],
        dtype=np.float64,
    )

def get_orientation(
    vertical_axis=1,
    positive_normal=True,
):
    """
    Return a Sionna radio-map orientation whose normal is aligned
    with the selected vertical axis.

    vertical_axis:
        0 -> X is height, map lies in YZ
        1 -> Y is height, map lies in XZ
        2 -> Z is height, map lies in XY
    """
    if vertical_axis == 0:
        # Rotate local +Z toward ±X around Y.
        angle = np.pi / 2 if positive_normal else -np.pi / 2
        return [0.0, angle, 0.0]

    if vertical_axis == 1:
        # Rotate local +Z toward ±Y around X.
        angle = -np.pi / 2 if positive_normal else np.pi / 2
        return [0.0, 0.0, angle]

    if vertical_axis == 2:
        # The default plane already lies in XY.
        if positive_normal:
            return [0.0, 0.0, 0.0]

        # Same plane, but normal reversed.
        return [0.0, np.pi, 0.0]

    raise ValueError("vertical_axis must be 0, 1, or 2")

def get_horizontal_axes(
    dimensions,
    vertical_axis=1
):
    if vertical_axis not in (0, 1, 2):
        raise ValueError("vertical_axis must be 0, 1, or 2")

    horizontal_axes = [
        axis for axis in range(3)
        if axis != vertical_axis
    ]

    return dimensions[horizontal_axes]

def get_scene_params(
    scene,
    vertical_axis=1,
    positive_normal=True,
    plane_elevation=None,
):
    bbox = scene.mi_scene.bbox()

    bbox_min = point3_to_numpy(bbox.min)
    bbox_max = point3_to_numpy(bbox.max)

    dimensions = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0

    if plane_elevation is None:
        plane_elevation = bbox_min[vertical_axis]

    plane_center = bbox_center.copy()
    plane_center[vertical_axis] = plane_elevation

    return {
        "min": bbox_min,
        "max": bbox_max,
        "bbox_center": bbox_center,
        "plane_center": plane_center,
        "dimensions": get_horizontal_axes(
            dimensions,
            vertical_axis,
        ),
        "orientation": get_orientation(
            vertical_axis,
            positive_normal,
        ),
        "plane_elevation": float(plane_elevation),
    }

def path_confing(
    max_depth=5,
    samples_per_src=10**6,
    max_num_paths_per_src=10**4,
    synthetic_array=True,
    los=True,
    specular_reflection=True,
    diffuse_reflection=True,
    refraction=True,
    diffraction=False,
    normalize_delays=False
):

    return {
        "max_depth": max_depth,
        "samples_per_src": samples_per_src,
        "max_num_paths_per_src": max_num_paths_per_src,
        "synthetic_array": synthetic_array,
        "los": los,
        "specular_reflection": specular_reflection,
        "diffuse_reflection": diffuse_reflection,
        "refraction": refraction,
        "diffraction": diffraction,
        "normalize_delays": normalize_delays
    }

def radiomap_config(
    scene,
    max_depth=5,
    cell_size=[3, 3],
    samples_per_tx=10**7,
    vertical_axis=1,
    positive_nornal=True
):
    
    params = get_scene_params(scene, vertical_axis, positive_nornal)

    return {
        "max_depth": int(max_depth),
        "cell_size": np.asarray(cell_size).tolist(),
        "center": np.asarray(params["plane_center"]).tolist(),
        "size": np.asarray(params["dimensions"]).tolist(),
        "orientation": np.asarray(params["orientation"]).tolist(),
        "samples_per_tx": int(samples_per_tx)
    }