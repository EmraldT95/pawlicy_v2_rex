import pybullet as p
import pybullet_data as pd
import numpy as np
from perlin_noise import PerlinNoise
import random

class TerrainConstants:
    """Some useful constant values for the robot and the terrain"""

    BASE_ORIENTATION = [0, 0, 0, 1]
    TYPES = ['mounts', 'maze', 'random', 'plane']
    FLAG_TO_FILENAME = {
        'mounts': "heightmaps/wm_height_out.png",
        'maze': "heightmaps/Maze.png"
    }
    ROBOT_INIT_POSITION = {
        'mounts': [0, 0, 0.32],
        'plane': [0, 0, 0.30],
        # 'hills': [0, 0, 1.98],
        'maze': [0, 0, 0.32],
        'random': [0, 0, 0.42]
    }
    TERRAIN_INIT_POSITION = {
        'mounts': [0, 0, 0.44],
        'plane': [0, 0, 0],
        # 'hills': [0, 0, 1.98],
        'maze': [0, 0, 0.04],
        'random': [0, 0, 0]
    }
    MESH_SCALES = {
        'mounts': [.1, .1, 8],
        'maze': [.3, .3, .1],
        'random': [.1, .1, .3]
    }

    def __init__(self):
        pass


class Terrain:
    """Creates/Adds terrains to the world"""

    def __init__(self, pybullet_client, type="random", columns=256, rows=256):
        self._pybullet_client = pybullet_client
        self._type = type
        self._columns = columns
        self._rows = rows
    
    def generate_terrain(self):
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())

        # Check whether we want one the pre-existing terrain data in pybullet or to generate a random one
        if self._type == "random":
            terrain_data = [0] * self._columns * self._rows
            noise = PerlinNoise(octaves=25, seed=random.random())
            for j in range(int(self._columns / 2)):
                for i in range(int(self._rows / 2)):
                    height = noise([j/self._columns,i/self._rows]) # Creates Perlin noise for smooth terrain generation
                    # height = random.uniform(0, height_perturbation_range)
                    terrain_data[2 * i + 2 * j * self._rows] = height
                    terrain_data[2 * i + 1 + 2 * j * self._rows] = height
                    terrain_data[2 * i + (2 * j + 1) * self._rows] = height
                    terrain_data[2 * i + 1 + (2 * j + 1) * self._rows] = height

            # Create the terrain in the simulator
            flags = p.GEOM_CONCAVE_INTERNAL_EDGE
            terrain_shape = self._pybullet_client.createCollisionShape(
                shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=TerrainConstants.MESH_SCALES[self._type],
                flags=flags,
                heightfieldTextureScaling=(self._rows - 1) / 2,
                heightfieldData=terrain_data,
                numHeightfieldRows=self._rows,
                numHeightfieldColumns=self._columns)
            terrain_id = self._pybullet_client.createMultiBody(0, terrain_shape)

        # For plane, just load the URDF file
        elif self._type == "plane":
            terrain_id = self._pybullet_client.loadURDF("plane_implicit.urdf")

        # This is mounts and maze types
        else:
            file_location = TerrainConstants.FLAG_TO_FILENAME[self._type]
            if not file_location:
                raise ValueError("Terrain of type %s was not found." % self._type)
            else:
                terrain_shape = self._pybullet_client.createCollisionShape(
                    shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
                    meshScale=TerrainConstants.MESH_SCALES[self._type],
                    fileName=file_location)
                terrain_id = self._pybullet_client.createMultiBody(0, terrain_shape) # Mass, Unique ID from createCollisionShape
                # For the "mounts" terrain type, pybullet has a texture file
                if self._type == "mounts":
                    textureId = self._pybullet_client.loadTexture("heightmaps/gimp_overlay_out.png")
                    self._pybullet_client.changeVisualShape(terrain_id, -1, textureUniqueId=textureId, rgbaColor=[1, 1, 1, 1])
                else:
                    self._pybullet_client.changeVisualShape(terrain_id, -1, rgbaColor=[0, 1, 1, 1])

        # Set the position and orientation of the terrain in the simulator
        self._pybullet_client.resetBasePositionAndOrientation(terrain_id, TerrainConstants.TERRAIN_INIT_POSITION[self._type], TerrainConstants.BASE_ORIENTATION)
        return terrain_id, self._type


class TerrainRandomizer:
    """Randomly generates a terrain for the gym envrionment"""

    def __init__(self, pybullet_client, columns=256, rows=256):
        self._pybullet_client = pybullet_client
        self._types = np.array(TerrainConstants.TYPES)
        self._columns = columns
        self._rows = rows

    def randomize(self):
        terrainType = np.random.choice(self._types, 1)[0]
        terrain = Terrain(pybullet_client=self._pybullet_client,
            type=terrainType,
            columns=self._columns,
            rows=self._rows)
        terrain_id, terrain_type = terrain.generate_terrain()
        return terrain_id, terrain_type
