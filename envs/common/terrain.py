# terrain generator
# from isaacgym.terrain_utils import *
import numpy as np
from collections.abc import Mapping
from typing import Union
from scipy import interpolate

from isaacgym import gymapi
import torch

# from isaacgym.terrain_utils import (
    # random_uniform_terrain,
    # pyramid_sloped_terrain,
    # discrete_obstacles_terrain,
    # pyramid_stairs_terrain,
    # stepping_stones_terrain,
    # convert_heightfield_to_trimesh,
    # SubTerrain,
# )

class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)

def random_uniform_terrain(terrain: SubTerrain, min_height, max_height, step=1, downsampled_scale=None,platform_size=1.0):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)
        platform_size (float): size of the flat platform at the center of the terrain [meters]

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    platform_size = int(platform_size / terrain.horizontal_scale)


    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)

    ratio = terrain.horizontal_scale / downsampled_scale
    height_field_downsampled = np.random.choice(
        heights_range, 
        (int(terrain.width * ratio), 
            int(terrain.length * ratio)))
    

    height_field_downsampled[int((terrain.width-platform_size) * ratio/2):int((terrain.width+platform_size) * ratio/2),
                            int((terrain.length-platform_size) * ratio/2):int((terrain.length+platform_size) * ratio/2)]=0


    x = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[1])

    f= interpolate.RectBivariateSpline(x,y,height_field_downsampled)

    x_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    y_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    z_upsampled = np.rint(f(x_upsampled,y_upsampled))

    terrain.height_field_raw += z_upsampled.astype(np.int16).T
    return terrain



# def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.):
#     """
#     Generate a sloped terrain

#     Parameters:
#         terrain (terrain): the terrain
#         slope (int): positive or negative slope
#         platform_size (float): size of the flat platform at the center of the terrain [meters]
#     Returns:
#         terrain (SubTerrain): update terrain
#     """
#     x = np.arange(0, terrain.width)
#     y = np.arange(0, terrain.length)
#     center_x = int(terrain.width / 2)
#     center_y = int(terrain.length / 2)
#     xx, yy = np.meshgrid(x, y, sparse=True)
#     xx = (center_x - np.abs(center_x-xx)) / center_x
#     yy = (center_y - np.abs(center_y-yy)) / center_y
#     xx = xx.reshape(terrain.width, 1)
#     yy = yy.reshape(1, terrain.length)
#     max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
#     terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

#     platform_size = int(platform_size / terrain.horizontal_scale / 2)
#     x1 = terrain.width // 2 - platform_size
#     x2 = terrain.width // 2 + platform_size
#     y1 = terrain.length // 2 - platform_size
#     y2 = terrain.length // 2 + platform_size

#     min_h = min(terrain.height_field_raw[x1, y1], 0)
#     max_h = max(terrain.height_field_raw[x1, y1], 0)
#     terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
#     return terrain

def pyramid_sloped_terrain(terrain: SubTerrain, slope=1, platform_size=1.):
    """
    Generate a sloped terrain

    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    slope = int(slope / terrain.vertical_scale*terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    dtype=terrain.height_field_raw.dtype
    x = np.empty(terrain.length,dtype=dtype)
    y = np.empty(terrain.width,dtype=dtype)

    xa = (min(terrain.length,terrain.width)-platform_size)//2+1
    xb = terrain.length - xa
    ya = xa
    yb = terrain.width - ya

    ha = np.arange(xa)# // step_width*step_height
    x[0:xa] = ha[:]
    x[xa:xb] = ha[-1]
    x[xb:] = ha[::-1]

    y[0:ya] = ha[:]
    y[ya:yb] = ha[-1]
    y[yb:] = ha[::-1]
    xx,yy = np.meshgrid(x, y, indexing='xy', sparse=False, copy=True)
    terrain.height_field_raw+=np.minimum(xx,yy)*slope
    return terrain

def pyramid_stairs_terrain(terrain: SubTerrain, step_width, step_height, platform_size=1.):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    xa = (min(terrain.length,terrain.width)-platform_size)//2+1
    xb = terrain.length - xa
    ya = xa
    yb = terrain.width - ya

    ha = np.arange(xa)# // step_width*step_height
    x = np.empty(terrain.length, dtype=np.int16)
    y = np.empty(terrain.width, dtype=np.int16)
    x[0:xa] = ha[:]
    x[xa:xb] = ha[-1]
    x[xb:] = ha[::-1]

    y[0:ya] = ha[:]
    y[ya:yb] = ha[-1]
    y[yb:] = ha[::-1]
    xx,yy = np.meshgrid(x, y, indexing='xy', sparse=False, copy=True)
    terrain.height_field_raw +=np.minimum(xx,yy) // step_width*step_height

    return terrain

# def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
#     """
#     Generate a terrain with gaps

#     Parameters:
#         terrain (terrain): the terrain
#         max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
#         min_size (float): minimum size of a rectangle obstacle [meters]
#         max_size (float): maximum size of a rectangle obstacle [meters]
#         num_rects (int): number of randomly generated obstacles
#         platform_size (float): size of the flat platform at the center of the terrain [meters]
#     Returns:
#         terrain (SubTerrain): update terrain
#     """
#     # switch parameters to discrete units
#     max_height = int(max_height / terrain.vertical_scale)
#     min_size = int(min_size / terrain.horizontal_scale)
#     max_size = int(max_size / terrain.horizontal_scale)
#     platform_size = int(platform_size / terrain.horizontal_scale)

#     (i, j) = terrain.height_field_raw.shape
#     height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
#     width_range = range(min_size, max_size, 4)
#     length_range = range(min_size, max_size, 4)

#     for _ in range(num_rects):
#         width = np.random.choice(width_range)
#         length = np.random.choice(length_range)
#         start_i = np.random.choice(range(0, i-width, 4))
#         start_j = np.random.choice(range(0, j-length, 4))
#         terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

#     x1 = (terrain.width - platform_size) // 2
#     x2 = (terrain.width + platform_size) // 2
#     y1 = (terrain.length - platform_size) // 2
#     y2 = (terrain.length + platform_size) // 2
#     terrain.height_field_raw[x1:x2, y1:y2] = 0
#     return terrain

def discrete_obstacles_terrain(terrain: SubTerrain, max_height, min_size, max_size, num_rects, platform_size=1.):
    """
    Generate a terrain with gaps

    Parameters:
        terrain (terrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    rect_sizes = np.random.choice(np.arange(min_size, max_size,4), num_rects*2).reshape((num_rects,2))
    # np.random.randint(min_size, max_size,num_rects*2).reshape((num_rects,2))
    # start_ij = (np.random.randn(num_rects,2)*(np.array([[i,j]])- rect_sizes)).astype(np.int16)//4*4
    height_ranges = np.random.choice(height_range, num_rects)
    for k in range(num_rects):
        width = rect_sizes[k][0]
        length = rect_sizes[k][1]
        # start_i = np.random.choice(range(0, i-width, 4))
        # start_j = np.random.choice(range(0, j-length, 4))
        # start_i = start_ij[k][0]
        # start_j = start_ij[k][1]
        start_i = np.random.randint(0, i-width)//4*4
        start_j = np.random.randint(0, j-length)//4*4
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = height_ranges[k]
    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-10):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles



class Terrain:
    def __init__(self, cfg: Mapping, num_robots: int, device: Union[torch.device, str], gym, sim) -> None:
        """
        Initializes the Terrain object with the given configuration parameters.

        Parameters:
            cfg (Mapping): A dictionary containing the configuration parameters for the terrain.
            num_robots (int): The number of robots in the environment.
            device (Union[torch.device, str]): The device to use for computation.
            gym (GymApi): The GymApi object for creating the terrain.
            sim (gymapi.Sim): The GymApi simulation object.

        Returns:
            None
        """
        self.device = device

        self.terrain_type: str = cfg["terrainType"]
        self.horizontal_scale: float = cfg["horizontalScale"]
        self.vertical_scale: float = cfg["verticalScale"]
        self.border_size: float = cfg["borderSize"]
        self.env_length: float = cfg["mapLength"]
        self.env_width: float = cfg["mapWidth"]
        self.env_rows: int = cfg["numLevels"]
        self.env_cols: int = cfg["numTerrains"]
        self.difficulty_scale: float = cfg.get("difficultySale", 1.0)  # difficulty multiplier
        self.platform_size: float = cfg["platformSize"]

        self.stair_width: float = cfg["stair"]["width"]
        self.stair_height: float = cfg["stair"]["height"]

        self.discrete_height: float = cfg["discrete"]["height"]
        self.discrete_size: tuple[float, float] = cfg["discrete"]["size"]
        self.discrete_num_rects: int = cfg["discrete"]["num_rects"]

        self.uniform_height: float = cfg["uniform"]["height"]
        self.uniform_step: float = cfg["uniform"]["step"]
        self.uniform_downsample: float = cfg["uniform"]["downsampled_scale"]

        self.slope = cfg["slope"]

        proportions = cfg["terrainProportions"]
        if len(proportions) == 6:
            # original terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stone]
            p = proportions[:]
            proportions = [p[1], 0, 0, p[2], p[3], p[0], 0, p[4], p[5]]

            # [0] rough_up,   \⎽/  # [1] rough_down, /⎺\   # [2] rough_flat      ⎽
            # [3] stair_up,   \⎽/  # [4] stair_down, /⎺\   # [5] smooth_up,     \⎽/
            # [6] smooth_down,/⎺\  # [7] discrete,   ☵☷   # [8] stepping_stone ▦▦

        self.proportions = np.cumsum(proportions) / np.sum(proportions)

        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        # sub terrain env origin
        self.env_origins = torch.zeros((self.env_rows, self.env_cols, 3), dtype=torch.float, device=self.device)

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)  # raw height field

        if cfg["curriculum"]:  # generate height field
            self.curriculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows, randomize=False)
        else:
            self.curriculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows, randomize=True)

        def set_common_params(params, offset=-self.border_size):
            if offset:
                params.transform.p.x = offset
                params.transform.p.y = offset
                params.transform.p.z = 0.0
            params.static_friction = cfg["staticFriction"]
            params.dynamic_friction = cfg["dynamicFriction"]
            params.restitution = cfg["restitution"]

        if self.terrain_type == "trimesh":
            # # create trimesh
            self.vertices, self.triangles = convert_heightfield_to_trimesh(
                self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"]
            )
            params = gymapi.TriangleMeshParams()
            params.nb_vertices = self.vertices.shape[0]
            params.nb_triangles = self.triangles.shape[0]
            set_common_params(params)
            gym.add_triangle_mesh(sim, self.vertices.flatten(order='C'), self.triangles.flatten(order='C'), params)
        elif self.terrain_type == "heightfield":
            # create height field
            params = gymapi.HeightFieldParams()
            params.column_scale = self.horizontal_scale
            params.row_scale = self.horizontal_scale
            params.vertical_scale = self.vertical_scale
            params.nbRows = self.tot_cols
            params.nbColumns = self.tot_rows
            set_common_params(params)
            gym.add_heightfield(sim, self.height_field_raw.ravel('F'), params)
        elif self.terrain_type == "plane":
            params = gymapi.PlaneParams()
            params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            set_common_params(params, offset=0)
            gym.add_ground(sim, params)

        # to torch
        self.height_samples = torch.tensor(self.height_field_raw, device=self.device).view(self.tot_rows, self.tot_cols)
        self.height_sample_limit = (self.height_samples.shape[0] - 2, self.height_samples.shape[1] - 2)

    def curriculum(self, num_robots, num_terrains, num_levels, randomize):
        """
        Generates a heght field for the terrain based on the given parameters.

        Parameters:
            num_robots (int): The total number of robots.
            num_terrains (int): The total number of terrains.
            num_levels (int): The total number of levels.
            randomize (bool): Whether to randomize the level and choice.

        Returns:
            None
        """
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        proportions = np.round(self.proportions * self.env_cols).astype(int)
        for j in range(num_terrains):
            for i in range(num_levels):

                if randomize:
                    level = np.random.randint(0, self.env_rows)
                    choice = np.random.randint(0, num_terrains)
                else:
                    level = i
                    choice = j
                difficulty = (level + 1) / num_levels
                # difficulty = (i + np.random.uniform(low=0, high=0.5)) / num_levels
                # choice = j / num_terrains

                slope = self.slope * self.difficulty_scale * difficulty
                discrete_height = self.discrete_height * self.difficulty_scale * difficulty
                # stepping stone
                stone_size = (2 - 1.5 * difficulty) * self.difficulty_scale
                stone_distance = 0.1 * self.difficulty_scale * difficulty

                uniform_height = self.uniform_height * self.difficulty_scale * difficulty
                uniform_step = max(self.uniform_step * self.difficulty_scale, self.vertical_scale)
                # uniform_step = self.vertical_scale

                stair_width = self.stair_width
                stair_height = self.stair_height * self.difficulty_scale * difficulty
                # stair_height = (0.05 + 0.175 * difficulty) * self.difficulty_scale

                terrain = SubTerrain(
                    "terrain",
                    width=self.length_per_env_pixels,
                    length=self.width_per_env_pixels,
                    vertical_scale=self.vertical_scale,
                    horizontal_scale=self.horizontal_scale,
                )
                # [0] rough_up,      \⎽/
                # [1] rough_down,    /⎺\
                # [2] rough_flat      ⎽
                # [3] stair_up,      \⎽/
                # [4] stair_down,    /⎺\
                # [5] smooth_up,     \⎽/
                # [6] smooth_down,   /⎺\
                # [7] discrete,      ☵☷
                # [8] stepping_stone  ▦▦

                # terrain types: [rough_up, rough_down, rough_flat, stair_up,
                # stair_down, smooth_up, smooth_down, discrete, stepping_stone]
                if choice < proportions[0]:  # [0] rough_up,      \⎽/
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=self.platform_size)
                    random_uniform_terrain(terrain, -uniform_height, uniform_height, uniform_step, self.uniform_downsample,platform_size=self.platform_size)
                elif choice < proportions[1]:  # [1] rough_down,    /⎺\
                    pyramid_sloped_terrain(terrain, slope=-slope, platform_size=self.platform_size)
                    random_uniform_terrain(terrain, -uniform_height, uniform_height, uniform_step, self.uniform_downsample,platform_size=self.platform_size)
                elif choice < proportions[2]:  # [2] rough_flat      ⎽
                    random_uniform_terrain(terrain, -uniform_height, uniform_height, uniform_step, self.uniform_downsample,platform_size=self.platform_size)
                elif choice < proportions[3]:  # [3] stair_up,      \⎽/
                    pyramid_stairs_terrain(
                        terrain, step_width=stair_width, step_height=-stair_height, platform_size=self.platform_size
                    )
                elif choice < proportions[4]:  # [4] stair_down,    /⎺\
                    pyramid_stairs_terrain(
                        terrain, step_width=stair_width, step_height=stair_height, platform_size=self.platform_size
                    )
                elif choice < proportions[5]:  # [5] smooth_up,     \⎽/
                    pyramid_sloped_terrain(terrain, slope=-slope, platform_size=self.platform_size)
                elif choice < proportions[6]:  # [6] smooth_down,   /⎺\
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=self.platform_size)
                elif choice < proportions[7]:  # [7] discrete,      ☵☷
                    # random_uniform_terrain(terrain, -uniform_height, uniform_height, uniform_step, self.uniform_downsample)
                    discrete_obstacles_terrain(
                        terrain,max_height=discrete_height,min_size=self.discrete_size[0], max_size=self.discrete_size[1], 
                        num_rects=self.discrete_num_rects, platform_size=self.platform_size)
                else:  # [8] stepping_stone  ▦▦
                    stepping_stones_terrain(
                        terrain,
                        stone_size=stone_size,
                        stone_distance=stone_distance,
                        max_height=0.0,
                        platform_size=self.platform_size,
                        depth=-1,
                    )

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map += 1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length / 2.0 - 1) / self.horizontal_scale)
                x2 = int((self.env_length / 2.0 + 1) / self.horizontal_scale)
                y1 = int((self.env_width / 2.0 - 1) / self.horizontal_scale)
                y2 = int((self.env_width / 2.0 + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
                self.env_origins[i, j] = torch.tensor([env_origin_x, env_origin_y, env_origin_z])

    def get_heights(self, points: torch.Tensor):
        points_ = ((points + self.border_size) / self.horizontal_scale).long()
        px = torch.clip(points_[..., 0].view(-1), 0, self.height_sample_limit[0])
        py = torch.clip(points_[..., 1].view(-1), 0, self.height_sample_limit[1])
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2) * self.vertical_scale
        return heights
