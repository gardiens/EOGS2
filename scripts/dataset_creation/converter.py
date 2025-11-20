import json
import numpy as np
import rpcm
import utm
from plyfile import PlyData, PlyElement


def ecef_to_latlon(x, y, z):
    """Convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)"""
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a**2
    esq = e**2
    b = np.sqrt(asq * (1 - esq))
    bsq = b**2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt((x**2) + (y**2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + (ep**2) * b * (np.sin(th) ** 3)), (p - esq * a * (np.cos(th) ** 3))
    )
    N = a / (np.sqrt(1 - esq * (np.sin(lat) ** 2)))
    alt = p / np.cos(lat) - N
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    return lat, lon, alt


def open_json_file(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)
    return file


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "f4"),
        ("green", "f4"),
        ("blue", "f4"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


class MyConverter:
    """This class is used to convert from LONLAT to a choosen world coordinate system. Base class.

    In the current implementation, the world coordinate system is a normalized ECEF coordinate
    system.
    """

    def __init__(self, scene_metadatas) -> None:
        # scene_metadatas = list of metadata dictionaries, one for each image in the scene.
        # We use all the different rpc models to compute the normalization factor for the conversion.
        # In this way, the conversion is the same for all images in the scene.

        # self.lonlat2ecef = Transformer.from_crs(
        #     "epsg:4326", "epsg:4978", always_xy=True
        # )

        vertices_UTM = []
        vertices_UTM_ground = []
        if len(scene_metadatas) == 0:
            print("Warning: scene_metadatas is empty")
        for metadata in scene_metadatas:
            rpc = rpcm.RPCModel(d=metadata["rpc"], dict_format="rpcm")
            width = metadata["width"]
            height = metadata["height"]
            min_altitude = metadata["min_alt"]
            max_altitude = metadata["max_alt"]
            for u in [0, width - 1]:
                for v in [0, height - 1]:
                    for a in [min_altitude, max_altitude]:
                        lon, lat = rpc.localization(u, v, a)

                        x, y, n, m = utm.from_latlon(lat, lon)
                        vertices_UTM.append(np.stack([x, y, a], axis=-1))
                    for a in [0.0]:
                        lon, lat = rpc.localization(u, v, a)
                        x, y, n, m = utm.from_latlon(lat, lon)
                        vertices_UTM_ground.append(np.stack([x, y, a], axis=-1))
        vertices_UTM = np.array(vertices_UTM)
        vertices_UTM_ground = np.array(vertices_UTM_ground)

        # Store the center of the scene in ECEF and recompute it in LONLAT
        self.centerofscene_UTM = vertices_UTM_ground.mean(axis=0)

        # Store the normalization
        self.shift = self.centerofscene_UTM

        self.n = n
        self.l = m

        # Search for the furthest point between vertices_ECEF and the center of the scene
        max_dist = 0
        for v in vertices_UTM:
            max_dist = max(max_dist, np.linalg.norm(v - self.centerofscene_UTM))
        self.scale = max_dist  # Remember to keep the same scale for all axes (as UTM is a euclidean coordinate system)

        # Compute the bounding box of the scene in the world coordinates
        vertices_world = (vertices_UTM - self.shift) / self.scale
        self.min_world = vertices_world.min(axis=0)
        self.max_world = vertices_world.max(axis=0)

        print("shift", self.shift, "scale", self.scale)
        print("min_world", self.min_world)
        print("max_world", self.max_world)
        print("volume", np.prod(self.max_world - self.min_world))

    # def _LONLAT2NormalizedECEF(self, LONLAT):
    #     X, Y, Z = self.lonlat2ecef.transform(
    #         LONLAT[..., 0], LONLAT[..., 1], LONLAT[..., 2]
    #     )
    #     ECEF = np.stack([X, Y, Z], axis=-1)
    #     return (ECEF - self.shift) / self.scale

    def _LONLAT2NormalizedUTM(self, LONLAT):
        # To UTM
        Z = LONLAT[..., 2]
        X, Y, _, _ = utm.from_latlon(LONLAT[..., 1], LONLAT[..., 0])
        UTM = np.stack([X, Y, Z], axis=-1)
        # Return normalized
        return (UTM - self.shift) / self.scale

    # def _ECEF2world(self, ECEF):
    #     normalizedECEF = (ECEF - self.shift) / self.scale
    #     return np.einsum("ij,...j->...i", self.R, normalizedECEF)

    def LONLAT2world(self, LONLAT):
        # Convert LONLAT to world coordinates
        # LONLAT is a numpy array of shape (N, 3) where N is the number of points
        # The output is a numpy array of shape (N, 3) with the world coordinates
        # Project to utm and normalized
        return self._LONLAT2NormalizedUTM(LONLAT)

    def transform_pcd(self, path_ply):
        # We suppose the path_ply to be in ECEF coordinates, then we convert it to the world coordinate system.
        pcd = PlyData.read(path_ply)
        vertices = pcd["vertex"]
        print("transformed pcd")
        positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T  #   Nx3
        print("position.shape", positions.shape)
        # the ply is supposed to be in ECEF coordinates.
        # convert the positon to latlon
        lat, lon, alt = ecef_to_latlon(
            positions[:, 0], positions[:, 1], positions[:, 2]
        )
        # convert the latlon to world coordinates
        LONLAT = np.stack([lon, lat, alt], axis=-1)
        position_utm = self.LONLAT2world(LONLAT)

        print("position_utm.shape", position_utm.shape)
        print("position_utm", position_utm)
        # transform back the pcd

        vertices["x"] = position_utm[:, 0]
        vertices["y"] = position_utm[:, 1]
        vertices["z"] = position_utm[:, 2]

        # for EOGS to work ,we need to set the colors and normals as well
        rgb = np.full((len(positions), 3), 1.1)
        return position_utm, rgb  # return the transformed pcd


class PanConverter(MyConverter):
    """Added PAN support."""

    def __init__(self, scene_metadatas, MSI_DIR, PAN_DIR, change_resolution) -> None:
        self.msi_dir = MSI_DIR
        self.change_resolution = change_resolution  # if True, we wil lchange the resolution based on the true image.
        self.pan_dir = PAN_DIR
        super().__init__(scene_metadatas)
