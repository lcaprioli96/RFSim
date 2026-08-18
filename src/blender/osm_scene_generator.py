import bpy
import os
import json

from mathutils import Vector


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = r"C:\Users\ludav\workshop\CNR\RFSim"
SCENE_NAME = "scene"

GROUND_MARGIN = 1.0       # meters around the imported buildings
GROUND_OFFSET = 0.05       # put ground slightly below buildings

CLEAR_SCENE = True


# ============================================================
# SCENE UTILITIES
# ============================================================

def clear_scene():
    """Remove every object currently in the Blender scene."""

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    print("Scene cleared.")


def coordinates_to_bbox(points):
    """
    Convert four (latitude, longitude) coordinates into the
    rectangular extent expected by Blosm.
    """

    if len(points) != 4:
        raise ValueError("Exactly four coordinates are required.")

    latitudes = [p[0] for p in points]
    longitudes = [p[1] for p in points]

    return {
        "min_lat": min(latitudes),
        "max_lat": max(latitudes),
        "min_lon": min(longitudes),
        "max_lon": max(longitudes),
    }


# ============================================================
# OSM IMPORT
# ============================================================

def import_osm_buildings(points):
    """
    Import OpenStreetMap buildings with Blosm.

    Returns:
        list[bpy.types.Object]:
            Mesh objects created by the import.
    """

    if not hasattr(bpy.context.scene, "blosm"):
        raise RuntimeError(
            "Blosm is not available. "
            "Install and enable the Blosm addon first."
        )

    bbox = coordinates_to_bbox(points)

    print("OSM bounding box:")
    print(f"  Latitude:  {bbox['min_lat']} -> {bbox['max_lat']}")
    print(f"  Longitude: {bbox['min_lon']} -> {bbox['max_lon']}")

    # Save current objects so that we can identify the newly imported ones.
    objects_before = set(bpy.data.objects)

    config = bpy.context.scene.blosm

    # --------------------------------------------------------
    # OSM source
    # --------------------------------------------------------

    config.dataType = "osm"
    config.osmSource = "server"

    # --------------------------------------------------------
    # Geographic extent
    # --------------------------------------------------------

    config.minLat = bbox["min_lat"]
    config.maxLat = bbox["max_lat"]

    config.minLon = bbox["min_lon"]
    config.maxLon = bbox["max_lon"]

    # --------------------------------------------------------
    # Import configuration
    # --------------------------------------------------------

    config.mode = "3Dsimple"

    config.buildings = True

    # Disable everything we don't currently need.
    config.water = False
    config.forests = False
    config.vegetation = False
    config.highways = False
    config.railways = False

    # IMPORTANT:
    # Keep buildings separate.
    # This will later allow material assignment building by building.
    config.singleObject = False

    config.relativeToInitialImport = False

    if hasattr(config, "commandLineMode"):
        config.commandLineMode = bpy.app.background

    print("Importing OSM buildings...")

    bpy.ops.blosm.import_data()

    # --------------------------------------------------------
    # Find objects created by Blosm
    # --------------------------------------------------------

    objects_after = set(bpy.data.objects)

    new_objects = objects_after - objects_before

    building_objects = [
        obj
        for obj in new_objects
        if obj.type == "MESH"
    ]

    if not building_objects:
        raise RuntimeError(
            "OSM import finished but no mesh buildings were found."
        )

    print(f"Imported {len(building_objects)} mesh objects.")

    return building_objects


# ============================================================
# BOUNDING BOX
# ============================================================

def get_objects_world_bounds(objects):
    """
    Compute world-space bounds of a collection of Blender objects.

    Returns:
        min_corner, max_corner
    """

    world_points = []

    for obj in objects:

        for corner in obj.bound_box:
            corner_world = obj.matrix_world @ Vector(corner)
            world_points.append(corner_world)

    if not world_points:
        raise ValueError("Cannot compute bounds of an empty object list.")

    min_corner = Vector((
        min(p.x for p in world_points),
        min(p.y for p in world_points),
        min(p.z for p in world_points),
    ))

    max_corner = Vector((
        max(p.x for p in world_points),
        max(p.y for p in world_points),
        max(p.z for p in world_points),
    ))

    return min_corner, max_corner


# ============================================================
# GROUND
# ============================================================

def create_ground_plane(building_objects, margin=10.0, offset=0.05):
    """
    Create a ground plane covering the entire imported scene.
    """

    bbox_min, bbox_max = get_objects_world_bounds(building_objects)

    width = bbox_max.x - bbox_min.x
    depth = bbox_max.y - bbox_min.y

    center_x = (bbox_min.x + bbox_max.x) / 2
    center_y = (bbox_min.y + bbox_max.y) / 2

    # Buildings imported without terrain normally start around z = 0.
    ground_z = bbox_min.z - offset

    bpy.ops.mesh.primitive_plane_add(
        size=2.0,
        location=(center_x, center_y, ground_z),
    )

    ground = bpy.context.object
    ground.name = "ground"

    ground.scale.x = width / 2 + margin
    ground.scale.y = depth / 2 + margin

    # Bake the scale into the mesh.
    bpy.ops.object.transform_apply(
        location=False,
        rotation=False,
        scale=True,
    )

    print(
        f"Ground created: "
        f"{width + 2 * margin:.2f} x "
        f"{depth + 2 * margin:.2f} m"
    )

    return ground


# ============================================================
# MATERIALS
# ============================================================

def create_material(name, base_color, roughness=0.8):
    """
    Create or update a simple Principled BSDF Blender material.

    base_color:
        RGB tuple with values in [0, 1]
    """

    material = bpy.data.materials.get(name)

    if material is None:
        material = bpy.data.materials.new(name=name)

    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    bsdf.inputs["Base Color"].default_value = (
        base_color[0],
        base_color[1],
        base_color[2],
        1.0,
    )

    bsdf.inputs["Roughness"].default_value = roughness

    links.new(
        bsdf.outputs["BSDF"],
        output.inputs["Surface"],
    )

    return material


def assign_material(obj, material):
    """Replace all existing materials on an object."""

    if obj.type != "MESH":
        return

    obj.data.materials.clear()
    obj.data.materials.append(material)


def replace_scene_materials(buildings, ground):
    """
    First simple material strategy:
        - all buildings -> concrete
        - ground -> ground material

    This function is intentionally separate because later we can replace
    it with OSM-tag-based material classification.
    """

    building_material = create_material(
        name="building_concrete",
        base_color=(0.55, 0.55, 0.55),
        roughness=0.8,
    )

    ground_material = create_material(
        name="ground_material",
        base_color=(0.20, 0.25, 0.15),
        roughness=1.0,
    )

    for building in buildings:
        assign_material(
            building,
            building_material,
        )

    assign_material(
        ground,
        ground_material,
    )

    print(
        f"Assigned building material to "
        f"{len(buildings)} buildings."
    )


# ============================================================
# MITSUBA EXPORT
# ============================================================

def export_mitsuba(root_dir, scene_name):
    """
    Export the complete scene using the Mitsuba Blender addon.
    """

    export_dir = os.path.join(root_dir, scene_name)

    os.makedirs(export_dir, exist_ok=True)

    xml_path = os.path.join(
        export_dir,
        f"{scene_name}.xml",
    )

    print(f"Exporting Mitsuba scene to:\n{xml_path}")

    try:
        bpy.ops.export_scene.mitsuba(
            filepath=xml_path,
            use_selection=False,
            split_files=False,
            export_ids=True,
            ignore_background=True,
        )

    except AttributeError as exc:
        raise RuntimeError(
            "Mitsuba Blender exporter is not available. "
            "Install and enable the Mitsuba Blender addon."
        ) from exc

    print("Mitsuba export completed.")

    return xml_path


# ============================================================
# PIPELINE
# ============================================================

def build_osm_scene(
    root_dir,
):
    """
    Complete pipeline:

    GPS coordinates
        ↓
    OSM import
        ↓
    building meshes
        ↓
    ground plane
        ↓
    material replacement
        ↓
    Mitsuba XML
    """
    with open(os.path.join(root_dir, "osm_coordinates.json"), "r") as f:
        config = json.load(f)

    scene_dir = os.path.join(ROOT_DIR, "scene")

    for scene in config["scene"]:

        scene_id = scene["scene_id"]
        points = scene["points"]
        if CLEAR_SCENE:
            clear_scene()

        # --------------------------------------------------------
        # 1. Import buildings
        # --------------------------------------------------------

        buildings = import_osm_buildings(points)

        # --------------------------------------------------------
        # 2. Create ground
        # --------------------------------------------------------

        ground = create_ground_plane(
            buildings,
            margin=GROUND_MARGIN,
            offset=GROUND_OFFSET,
        )

        # --------------------------------------------------------
        # 3. Replace materials
        # --------------------------------------------------------

        replace_scene_materials(
            buildings,
            ground,
        )

        # --------------------------------------------------------
        # 4. Export Mitsuba scene
        # --------------------------------------------------------

        xml_path = export_mitsuba(
            scene_dir,
            scene_id,
        )

        print("\nScene creation completed.")
        print(f"Buildings: {len(buildings)}")
        print(f"Mitsuba XML: {xml_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = build_osm_scene(
        ROOT_DIR,
    )