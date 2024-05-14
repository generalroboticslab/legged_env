import trimesh
import urdfpy
import numpy as np


def get_urdf_scene(urdf:urdfpy.URDF,cfg=None,use_collision=True):
    """Visualize the URDF in a given configuration.
    Parameters
    ----------
    cfg : dict or (n), float
        A map from joints or joint names to configuration values for
        each joint, or a list containing a value for each actuated joint
        in sorted order from the base link.
        If not specified, all joints are assumed to be in their default
        configurations.
    use_collision : bool
        If True, the collision geometry is visualized instead of
        the visual geometry.
    """
    fk = urdf.collision_trimesh_fk()
    if use_collision:
        fk = urdf.collision_trimesh_fk(cfg=cfg)
    else:
        fk = urdf.visual_trimesh_fk(cfg=cfg)

    scene = trimesh.scene.Scene()
    for tm in fk:
        pose = fk[tm]
        mesh = tm
        scene.add_geometry(mesh, transform=pose)
    return scene


def get_urdf_bounding_box(urdf:urdfpy.URDF):
    """return urdf bounding box
    Example:
        urdf_path=....
        urdf = urdfpy.URDF.load(urdf_path)
        bounding_box = get_urdf_bounding_box(urdf)
    """
    scene = get_urdf_scene(urdf)
    bounding_box = scene.bounding_box
    return bounding_box