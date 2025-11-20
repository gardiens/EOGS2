import torch
import torch.nn.functional as F
import numpy as np
import iio
import einops
import os
import json
from tqdm import tqdm
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)
from eval.eval_dsm import main_hydra_dsm
try:
    import clearml
    try:
        from clearml_utils import safe_resume_clearml, safe_init_clearml
    except:
        from utils.clearml_utils import safe_resume_clearml, safe_init_clearml
    CLEARML_FOUND = True 
except:
    CLEARML_FOUND = False

def cubify(
    voxels_values: torch.FloatTensor, voxels_vertices: torch.FloatTensor, thresh: float
):
    """Vectorized “cubify” in PyTorch: each occupied voxel (value ≥ thresh) becomes a cube.
    Internal faces (shared by two occupied voxels) are omitted.  Returns a mesh (vertices, faces).

    Args:
        voxels_values: FloatTensor of shape (X, Y, Z).  A voxel is “occupied” if ≥ thresh.
        voxels_vertices: FloatTensor of shape (X+1, Y+1, Z+1, 3).  The XYZ coordinates
                         of each grid vertex.  Voxel (i,j,k) spans the 8 corners
                         voxels_vertices[i:i+2, j:j+2, k:k+2, :].
        thresh: float threshold.

    Returns:
        vertices: FloatTensor of shape (NV, 3).  The XYZ positions of all unique exterior
                  grid-vertices used by the mesh.
        faces:    LongTensor of shape (NF, 3).  Each row is a triangle (three indices into vertices).
    """
    # 1) occupancy mask
    occ = voxels_values >= thresh  # bool tensor, shape (X,Y,Z)
    X, Y, Z = occ.shape
    if not occ.any():
        return torch.empty((0, 3), dtype=voxels_vertices.dtype), torch.empty(
            (0, 3), dtype=torch.long
        )

    # 2) build neighbor‐occupancy tensors (same shape as occ), treating out‐of‐bounds as False
    occ_xm1 = torch.zeros_like(occ)
    occ_xm1[1:, :, :] = occ[:-1, :, :]
    occ_xp1 = torch.zeros_like(occ)
    occ_xp1[:-1, :, :] = occ[1:, :, :]

    occ_ym1 = torch.zeros_like(occ)
    occ_ym1[:, 1:, :] = occ[:, :-1, :]
    occ_yp1 = torch.zeros_like(occ)
    occ_yp1[:, :-1, :] = occ[:, 1:, :]

    occ_zm1 = torch.zeros_like(occ)
    occ_zm1[:, :, 1:] = occ[:, :, :-1]
    occ_zp1 = torch.zeros_like(occ)
    occ_zp1[:, :, :-1] = occ[:, :, 1:]

    # 3) for each of the six directions, find the “exterior‐face mask”
    neg_x_mask = occ & (~occ_xm1)  # faces at x = i
    pos_x_mask = occ & (~occ_xp1)  # faces at x = i+1
    neg_y_mask = occ & (~occ_ym1)  # faces at y = j
    pos_y_mask = occ & (~occ_yp1)  # faces at y = j+1
    neg_z_mask = occ & (~occ_zm1)  # faces at z = k
    pos_z_mask = occ & (~occ_zp1)  # faces at z = k+1

    # 4) grid sizes for linear indexing
    Xg, Yg, Zg = X + 1, Y + 1, Z + 1
    # We will map a grid‐vertex (u,v,w)  →  linear index L = u*(Yg*Zg) + v*(Zg) + w

    # Helper to build quads for a given mask and offset patterns
    def build_quads(mask, i_offs, j_offs, k_offs):
        """Given mask of shape (X,Y,Z) and four offset-tuples (i_offs, j_offs, k_offs), each of
        shape (N,).

        Actually, i_offs, j_offs, k_offs are lists of length 4 giving relative offsets from (i,j,k)
        to each corner of the quad. We do this more directly by stacking broadcasted arithmetic.
        """
        idx = mask.nonzero(as_tuple=False)  # (N,3) tensor of (i,j,k)
        if idx.numel() == 0:
            return torch.empty((0, 4), dtype=torch.long)

        i = idx[:, 0]  # shape (N,)
        j = idx[:, 1]
        k = idx[:, 2]

        # For each of the 4 corners, we add the offsets and compute linear index.
        lin_indices = []
        for di, dj, dk in zip(i_offs, j_offs, k_offs):
            ui = i + di  # shape (N,)
            vj = j + dj
            wk = k + dk
            lin = ui * (Yg * Zg) + vj * Zg + wk  # shape (N,)
            lin_indices.append(lin)

        # Stack into (N,4)
        return torch.stack(lin_indices, dim=1)

    # 5) For each direction, specify the 4 corner‐offsets:
    #    (di, dj, dk) for each corner of the quad, in a consistent winding.

    # ——— -X face at x = i: corners (i,j,k), (i,j,k+1), (i,j+1,k+1), (i,j+1,k)
    quads_neg_x = build_quads(
        neg_x_mask, i_offs=[0, 0, 0, 0], j_offs=[0, 0, 1, 1], k_offs=[0, 1, 1, 0]
    )

    # # ——— +X face at x = i+1: corners (i+1,j,k), (i+1,j,k+1), (i+1,j+1,k+1), (i+1,j+1,k)
    # quads_pos_x = build_quads(
    #     pos_x_mask,
    #     i_offs=[1, 1, 1, 1],
    #     j_offs=[0, 0, 1, 1],
    #     k_offs=[0, 1, 1, 0]
    # )

    # ——— +X face at x = i+1: corners (i+1,j,k), (i+1,j+1,k), (i+1,j+1,k+1), (i+1,j,k+1)
    quads_pos_x = build_quads(
        pos_x_mask, i_offs=[1, 1, 1, 1], j_offs=[0, 1, 1, 0], k_offs=[0, 0, 1, 1]
    )

    # ——— -Y face at y = j: corners (i,j,k), (i+1,j,k), (i+1,j,k+1), (i,j,k+1)
    quads_neg_y = build_quads(
        neg_y_mask, i_offs=[0, 1, 1, 0], j_offs=[0, 0, 0, 0], k_offs=[0, 0, 1, 1]
    )

    # ——— +Y face at y = j+1: corners (i,j+1,k), (i,j+1,k+1), (i+1,j+1,k+1), (i+1,j+1,k)
    quads_pos_y = build_quads(
        pos_y_mask, i_offs=[0, 0, 1, 1], j_offs=[1, 1, 1, 1], k_offs=[0, 1, 1, 0]
    )

    # ——— -Z face at z = k: corners (i,j,k), (i,j+1,k), (i+1,j+1,k), (i+1,j,k)
    quads_neg_z = build_quads(
        neg_z_mask, i_offs=[0, 0, 1, 1], j_offs=[0, 1, 1, 0], k_offs=[0, 0, 0, 0]
    )

    # ——— +Z face at z = k+1: corners (i,j,k+1), (i+1,j,k+1), (i+1,j+1,k+1), (i,j+1,k+1)
    quads_pos_z = build_quads(
        pos_z_mask, i_offs=[0, 1, 1, 0], j_offs=[0, 0, 1, 1], k_offs=[1, 1, 1, 1]
    )

    # 6) Stack all quads into one (M,4) LongTensor
    all_quads = torch.cat(
        [quads_neg_x, quads_pos_x, quads_neg_y, quads_pos_y, quads_neg_z, quads_pos_z],
        dim=0,
    )  # shape (M,4)

    if all_quads.numel() == 0:
        # No exterior faces
        return torch.empty((0, 3), dtype=voxels_vertices.dtype), torch.empty(
            (0, 3), dtype=torch.long
        )

    # 7) Flatten to (4*M,), compute unique vertex‐indices, and the inverse map
    flat_quads = all_quads.view(-1)  # shape (4*M,)
    unique_vs, inv = torch.unique(flat_quads, sorted=True, return_inverse=True)
    # unique_vs: shape (NV,), dtype long
    # inv: shape (4*M,), dtype long, each entry is index into unique_vs

    # Re‐shape inv into (M,4) to get remapped quads
    quads_remapped = inv.view(-1, 4)  # shape (M,4)

    # 8) Split each quad into two triangles (v0,v1,v2) and (v0,v2,v3)
    tri1 = quads_remapped[:, [0, 1, 2]]  # shape (M,3)
    tri2 = quads_remapped[:, [0, 2, 3]]  # shape (M,3)
    faces = torch.cat([tri1, tri2], dim=0)  # shape (2*M, 3), dtype long

    # 9) Build the (NV,3) vertex‐coordinate tensor by indexing into the flattened grid‐vertex array
    #    First, flatten voxels_vertices from shape (X+1, Y+1, Z+1, 3) → ( (X+1)*(Y+1)*(Z+1), 3 )
    flat_verts = voxels_vertices.view(-1, 3)  # dtype = same as voxels_vertices

    # unique_vs gives the linear indices of the grid‐vertices we need; gather them
    vertices = flat_verts[unique_vs, :]  # shape (NV, 3)

    return vertices, faces


class RangeImageEOGS:
    def numpy_img_to_tensor(self, img):
        img = torch.tensor(img, dtype=torch.float32, device=self.device)
        assert len(img.shape) in [2, 3]
        if len(img.shape) == 2:
            img = img.unsqueeze(-1)

        img = img.permute(2, 0, 1)  # Change to (C, H, W)
        img = img.unsqueeze(0)  # Add batch dimension
        return img

    def __init__(
        self,
        metadata: dict,
        altitude_img: torch.Tensor,
    ):
        super().__init__()

        self.device = torch.device("cuda:0")

        self.img_name = metadata["img"]
        self.model_scale = metadata["model"]["scale"]
        # Affine model is a tuple of (coef, intercept)
        # Mapping a 3D point x=(x0,x1,x2) to a 2D point (u0,u1) (+altitude) using the affine model:
        # u = W @ x + w, where W is the coef matrix and w is the intercept vector.
        self.affine_model = (
            torch.tensor(
                metadata["model"]["coef_"], dtype=torch.float32, device=self.device
            ),
            torch.tensor(
                metadata["model"]["intercept_"], dtype=torch.float32, device=self.device
            ),
        )

        self.view_direction = torch.linalg.solve(
            self.affine_model[0], torch.tensor([0, 0, 1.0], device=self.device)
        )
        self.view_direction = torch.nn.functional.normalize(
            self.view_direction, dim=0, eps=1e-6
        )

        self.altitude_img = self.numpy_img_to_tensor(altitude_img)

        _, _, self.height, self.width = self.altitude_img.shape

        # Estimate the normal vector of the surface
        self.pixels_normals = self.reconstruct_normals()
        self.pixels_angle = einops.einsum(
            self.pixels_normals, -self.view_direction, "b c h w, c -> b h w"
        ).unsqueeze(1)

    def _world_to_view(self, x):
        return torch.nn.functional.linear(x, self.affine_model[0], self.affine_model[1])

    def _view_to_world(self, x):
        # x : (..., 3)
        Ainv = torch.linalg.inv(self.affine_model[0])
        Ainvb = Ainv @ self.affine_model[1]
        y = torch.nn.functional.linear(x, Ainv, -Ainvb)
        return y

    def reconstruct_normals(self) -> torch.Tensor:
        # These are the positions of the surfaces imagined by the pixels in "view" coordinates.
        u = torch.arange(self.width, dtype=torch.float32, device=self.device)
        v = torch.arange(self.height, dtype=torch.float32, device=self.device)
        U, V = torch.meshgrid(u, v, indexing="ij")
        UVA = torch.stack([U, V, self.altitude_img.squeeze().T], axis=-1)
        # This are the view/local/intrinsic coordinates of the image with U and V in [-1,1] and A in [min_altitude, max_altitude]
        view = (UVA + torch.tensor([0.5, 0.5, 0], device=self.device)) * torch.tensor(
            [1 / self.width, 1 / self.height, 1], device=self.device
        )
        view[..., :2] = view[..., :2] * 2 - 1

        # Convert view coordinates to world coordinates
        world_pos = self._view_to_world(view)
        world_pos = einops.rearrange(world_pos, "w h c -> 1 c h w")

        windows = F.unfold(
            world_pos, kernel_size=(5, 5), dilation=1, padding=2, stride=1
        )  # (1, 3*5*5, H*W)
        windows = einops.rearrange(
            windows,
            "1 (c k1 k2) (h w) -> 1 c h w k1 k2",
            k1=5,
            k2=5,
            c=3,
            h=self.height,
            w=self.width,
        )

        # | x_{-2} | x_{-1} | x_{0} | x_{1} | x_{2} |
        # Linear model from left: x_{0} \approx = x_{-2} + 2 * (x_{-1} - x_{-2})
        # Linear model from right: x_{0} \approx = x_{2} + 2 * (x_{1} - x_{2})
        pred_left_x = windows[..., 2, 0] + 2 * (windows[..., 2, 1] - windows[..., 2, 0])
        pred_right_x = windows[..., 2, 4] + 2 * (
            windows[..., 2, 3] - windows[..., 2, 4]
        )
        error_left_x = torch.linalg.vector_norm(pred_left_x - windows[..., 2, 2], dim=1)
        error_right_x = torch.linalg.vector_norm(
            pred_right_x - windows[..., 2, 2], dim=1
        )
        dx = torch.where(
            error_left_x < error_right_x,
            (windows[..., 2, 2] - windows[..., 2, 0]) * 0.5,
            (windows[..., 2, 4] - windows[..., 2, 2]) * 0.5,
        )

        pred_left_y = windows[..., 0, 2] + 2 * (windows[..., 1, 2] - windows[..., 0, 2])
        pred_right_y = windows[..., 4, 2] + 2 * (
            windows[..., 3, 2] - windows[..., 4, 2]
        )
        error_left_y = torch.linalg.vector_norm(pred_left_y - windows[..., 2, 2], dim=1)
        error_right_y = torch.linalg.vector_norm(
            pred_right_y - windows[..., 2, 2], dim=1
        )
        dy = torch.where(
            error_left_y < error_right_y,
            (windows[..., 2, 2] - windows[..., 0, 2]) * 0.5,
            (windows[..., 4, 2] - windows[..., 2, 2]) * 0.5,
        )

        # # Pad by 1 pixel on each side
        # padded = F.pad(world_pos, (1, 1, 1, 1), mode="replicate")  # (1, 3, H+2, W+2)

        # # Get neighbor positions for central differences
        # pos_left = padded[:, :, 1:-1, 0:-2]  # x-1
        # pos_right = padded[:, :, 1:-1, 2:]  # x+1
        # pos_up = padded[:, :, 0:-2, 1:-1]  # y-1
        # pos_down = padded[:, :, 2:, 1:-1]  # y+1

        # # Compute spatial derivatives
        # dx = (pos_right - pos_left) * 0.5  # shape: (1, 3, H, W)
        # dy = (pos_down - pos_up) * 0.5

        # Cross product gives the normal
        normals = torch.cross(dx, dy, dim=1)  # (1, 3, H, W)

        normals = F.normalize(normals, dim=1, eps=1e-6)
        return normals

    def get_weights(self) -> torch.Tensor:
        return self.pixels_angle.clamp(min=0.0, max=1.0)

    def sample_sdf(self, pts_world_coords):
        """Sample SDF values from the altitude image based on world coordinates.

        :param pts_world_coords: Tensor of shape (N, 3) representing points in world coordinates.
        :return: Tuple of (sdf_values, valid_mask, weights) where sdf_values is a tensor of shape
            (N,) representing SDF values, and valid_mask is a tensor of shape (N,) and weights is a
            tensor of shape (N,)
        """
        # given a point in world coordinates (x,y,z), we need to convert it to view coordinates (u,v,altitude).
        # Then we use torch.grid_sample to sample the altitude image at these coordinates (u,v).
        # So now we project back to world coordines (x,y,z) using the inverse of the affine model and the newly sampled altitude.
        # Hence, we get a new point in world coordinates (x',y',z').
        # Finally, we return the distance between the original point (x,y,z) and the new point (x',y',z').
        # TODO: for now, we assume that all points are valid

        pts_world_coords = (
            pts_world_coords / self.model_scale
        )  # Scale the points to the model scale
        pts_view_coords = self._world_to_view(pts_world_coords)
        # Sample altitude and normals

        features = torch.cat([self.altitude_img, self.get_weights()], dim=1)
        sampled_features = F.grid_sample(
            features,
            pts_view_coords[None, :, None, :2],
            mode="bilinear",
            align_corners=True,
        ).squeeze()
        altitude_values = sampled_features[0, :]
        weights = sampled_features[1, :]

        valid_mask = (pts_view_coords[:, :2].abs() <= 1.0).all(dim=1)

        # Compute SDF values
        pts_view_coords_new = pts_view_coords.clone()
        pts_view_coords_new[:, 2] = altitude_values
        pts_world_coords_new = self._view_to_world(pts_view_coords_new)

        # Compute SDF values
        distances = torch.linalg.norm(pts_world_coords_new - pts_world_coords, dim=1)
        distances = distances * torch.sign(pts_view_coords[:, 2] - altitude_values)
        distances = distances * self.model_scale

        return distances, valid_mask, weights


class TSDFVolume:
    """Volumetric with TSDF representation."""

    def __init__(
        self, vol_bounds: np.ndarray, vox_size: float, trunc_margin_fact: float
    ) -> None:
        """Constructor.

        :param vol_bounds: An ndarray is shape (3,2), define the min & max bounds of voxels.
        :param voxel_size: Voxel size in meters.
        """

        self.device = torch.device("cuda:0")
        self.vox_size = vox_size
        self._trunc_margin = trunc_margin_fact * self.vox_size  # truncation on SDF

        vol_bounds = torch.tensor(vol_bounds, dtype=torch.float32, device=self.device)
        assert vol_bounds.shape == (3, 2), "vol_bounds should be of shape (3,2)"

        # We now create a tensor for the voxel coordinates in the "world" coordinate system.
        self.num_voxels_per_dimension = (
            vol_bounds[:, 1] - vol_bounds[:, 0]
        ) // vox_size + 1
        self.num_voxels_per_dimension = self.num_voxels_per_dimension.ceil().long()
        starts = vol_bounds[:, 0]
        ends = vol_bounds[:, 0] + self.num_voxels_per_dimension * vox_size
        self.num_voxels_per_dimension = tuple(
            self.num_voxels_per_dimension.cpu().numpy().tolist()
        )

        print("Number of voxels per dimension: ", self.num_voxels_per_dimension)

        self.axes = [
            torch.linspace(
                starts[i], ends[i], self.num_voxels_per_dimension[i], device=self.device
            )
            for i in range(3)
        ]

        self.world_coords = (
            torch.stack(torch.meshgrid(*self.axes, indexing="ij"), dim=-1)
            .reshape(-1, 3)
            .to(self.device, torch.float32)
        )

        # We also need a tensor of indices for the voxel coordinates.
        self.vox_coords = torch.tensor(
            np.indices(self.num_voxels_per_dimension).reshape(3, -1).T
        ).to(self.device, torch.int64)

        # We also need a tensor of the vertices for the voxel coordinates.
        # This is a grid in one extra voxel per dimension and its it shifted by half a voxel size.
        starts = starts - vox_size / 2.0
        ends = ends + vox_size / 2.0
        self.voxels_vertices = torch.meshgrid(
            [
                torch.linspace(
                    starts[0], ends[0], self.num_voxels_per_dimension[0] + 1
                ),
                torch.linspace(
                    starts[1], ends[1], self.num_voxels_per_dimension[1] + 1
                ),
                torch.linspace(
                    starts[2], ends[2], self.num_voxels_per_dimension[2] + 1
                ),
            ],
            indexing="ij",
        )
        self.voxels_vertices = (
            torch.stack(
                (
                    self.voxels_vertices[0],
                    self.voxels_vertices[1],
                    self.voxels_vertices[2],
                ),
                dim=-1,
            ).to(self.device, torch.float32)
            # .reshape(-1, 3)
        )

        # TSDF & weights
        self._tsdf_vol = torch.ones(
            size=self.num_voxels_per_dimension, device=self.device, dtype=torch.float32
        )
        self._weight_vol = torch.zeros(
            size=self.num_voxels_per_dimension, device=self.device, dtype=torch.float32
        )

    def integrate(self, rangeimage: RangeImageEOGS):
        """Integrate an depth image to the TSDF volume.

        :param depth_img: depth image with depth value in meter.
        :param intrinsics: camera intrinsics of shape (3,3).
        :param cam_pose: camera pose, transform matrix of shape (4,4)
        :param weight: weight assign for current frame, higher value indicate higher confidence
        """

        # Compute and Integrate TSDF
        sdf_value, voxel_mask, weights = rangeimage.sample_sdf(self.world_coords)
        voxel_mask &= (
            sdf_value >= -self._trunc_margin
        )  # Only keep voxels within truncation margin
        # Truncate SDF

        tsdf_value = torch.minimum(
            torch.ones_like(sdf_value, device=self.device),
            sdf_value / self._trunc_margin,
        )
        tsdf_value = tsdf_value[voxel_mask]

        # Get coordinates of valid voxels with valid TSDF value
        valid_vox_x = self.vox_coords[voxel_mask, 0]
        valid_vox_y = self.vox_coords[voxel_mask, 1]
        valid_vox_z = self.vox_coords[voxel_mask, 2]

        # Update TSDF of cooresponding voxels
        weight_old = self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_old = self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]

        weights = weights.reshape(*self._weight_vol.shape)
        weights = weights[valid_vox_x, valid_vox_y, valid_vox_z]

        tsdf_new, weight_new = self.update_tsdf(
            tsdf_old, tsdf_value, weight_old, weights
        )

        self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_new
        self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = weight_new

    def update_tsdf(self, tsdf_old, tsdf_new, weight_old, obs_weight):
        """Update the TSDF value of given voxel V = (wv + WV) / w + W.

        :param tsdf_old: Old TSDF values.
        :param tsdf_new: New TSDF values.
        :param weight_old: Voxels weights.
        :param obs_weight: Weight of current update.
        :return: Updated TSDF values & Updated weights.
        """

        tsdf_vol_int = torch.empty_like(
            tsdf_old, dtype=torch.float32, device=self.device
        )
        weight_new = torch.empty_like(
            weight_old, dtype=torch.float32, device=self.device
        )

        weight_new = weight_old + obs_weight
        tsdf_vol_int = (weight_old * tsdf_old + obs_weight * tsdf_new) / weight_new

        return tsdf_vol_int, weight_new

    def extract_mesh(self, output_mesh_path):
        import mcubes

        vertices, triangles = mcubes.marching_cubes(self._tsdf_vol.cpu().numpy(), 0)
        # Save the mesh to an OBJ file
        print("Saving mesh to {}".format(output_mesh_path))
        mcubes.export_obj(vertices, triangles, output_mesh_path)

    def extract_dsm(self, scene_params, resolution, output_dir):
        """Get a point cloud expressed in UTM coordinates."""
        # The tensor self._tsdf_vol has shape ([x], [y], [z]) and contains the TSDF values.
        # We need to find, for each (x,y) pair, the first voxel with a TSDF value < 0.
        idx = torch.arange(0, self._tsdf_vol.shape[-1], device=self.device)
        V2 = (self._tsdf_vol < 0) * idx
        indices = torch.argmax(V2, dim=-1, keepdim=False)
        z_ax = self.axes[-1]
        assert indices.max() <= len(z_ax) - 1, "Indices out of bounds for z-axis"
        z_values = z_ax[indices]

        # return z_values
        xy_coords = torch.stack(
            torch.meshgrid([self.axes[0], self.axes[1]], indexing="ij"), dim=-1
        )

        cloud = (
            torch.cat(
                [
                    xy_coords,
                    z_values.unsqueeze(-1),
                ],
                dim=-1,
            )
            .detach()
            .cpu()
            .reshape(-1, 3)
            .numpy()
        )

        # Unnormalized the point cloud so we're in normal utm again
        # cloud = cloud * scene_params[1] + scene_params[0]
        cloud = cloud + scene_params[0]

        xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
        ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
        xoff = np.floor(xmin / resolution) * resolution
        xsize = int(1 + np.floor((xmax - xoff) / resolution))
        yoff = np.ceil(ymax / resolution) * resolution
        ysize = int(1 - np.floor((ymin - yoff) / resolution))

        # run plyflatten
        from plyflatten import plyflatten
        import rasterio
        from plyflatten import plyflatten
        from plyflatten.utils import rasterio_crs, crs_proj
        import affine

        dsm = plyflatten(
            cloud, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float("inf")
        )

        crs = rasterio_crs(
            crs_proj("{}{}".format(scene_params[2], scene_params[3]), crs_type="UTM")
        )

        profile = {}
        profile["dtype"] = dsm.dtype
        profile["height"] = dsm.shape[0]
        profile["width"] = dsm.shape[1]
        profile["count"] = 1
        profile["driver"] = "GTiff"
        profile["nodata"] = float("nan")
        profile["crs"] = crs
        profile["transform"] = affine.Affine(
            resolution, 0.0, xoff, 0.0, -resolution, yoff
        )
        with rasterio.open(os.path.join(output_dir, "dsm.iio"), "w", **profile) as f:
            f.write(dsm[:, :, 0], 1)

        return dsm

    def apply_prior(self):
        untouched_voxels = (self._weight_vol == 0) & (self._tsdf_vol == 1.0)
        occ = self._tsdf_vol <= 0

        # Set to occupied the all the voxels at lowest levels
        self._tsdf_vol[:, :, 0] = -1.0  # Set to occupied
        self._weight_vol[:, :, 0] = 1.0  # Set weight to 1.0

        # First, remove any 1x1x1 occupied but isolated voxels
        # We do this by checking if the occupied voxel has any neighbors that are also occupied.
        # We use a 3x3x3 kernel to check for neighbors.
        kernel = torch.ones((3, 3, 3), device=self.device, dtype=torch.float32)
        # Convolve the occupied volume with the kernel
        occ_conv = F.conv3d(
            occ.unsqueeze(0).unsqueeze(0).float(),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1,
        ).squeeze()
        # If a voxel is occupied and has no neighbors, we set it to untouched
        isolated_voxels = (occ_conv == 1) & occ.squeeze()
        self._tsdf_vol[isolated_voxels] = 1.0  # Set to untouched
        self._weight_vol[isolated_voxels] = 0.0  # Set weight to 0.0

        # If a untouched voxel is below an occupied voxel, we set it to occupied.
        # Get the highest occupied voxel in each column
        idx = torch.arange(0, self._tsdf_vol.shape[-1], device=self.device)
        V2 = occ * idx
        indices = torch.argmax(V2, dim=-1, keepdim=False)
        vox_idx = torch.tensor(np.indices(self.num_voxels_per_dimension)).to(
            self.device, torch.int64
        )
        mask = vox_idx[-1, :, :, :] < indices.unsqueeze(-1)
        mask = mask & untouched_voxels
        # Set the untouched voxels to occupied if they are below an occupied voxel
        self._tsdf_vol[mask] = -1.0  # Set to occupied
        self._weight_vol[mask] = 1.0  # Set weight to 1.0


def main(
    slanted_altitude_dir,
    scene_name,
    vox_size,
    trunc_margin_fact,
    output_dir,
    affine_models_json_path,
    train_txt_split_path,
    gt_dir,
    ref_key,
    cfg_rendering,
    iteration,
    export_mesh: bool,
    output_mesh_path: str,
):
    os.makedirs(output_dir, exist_ok=True)

    with open(affine_models_json_path, "r") as f:
        metadatas = json.load(f)
    with open(train_txt_split_path, "r") as f:
        train_split = f.readlines()
        train_split = sorted([line.strip() for line in train_split])

    # Define volume bounds and voxel size
    metadatas = metadatas[ref_key]
    model_scale = metadatas[0]["model"]["scale"]
    vol_bounds = np.array(
        [metadatas[0]["model"]["min_world"], metadatas[0]["model"]["max_world"]]
    ).T
    vol_bounds = vol_bounds * model_scale
    tsdf_vol = TSDFVolume(
        vol_bounds,
        vox_size=vox_size,
        trunc_margin_fact=trunc_margin_fact,
    )

    scene_params = [
        np.array(metadatas[0]["model"]["center"]),
        metadatas[0]["model"]["scale"],
        metadatas[0]["model"]["n"],
        metadatas[0]["model"]["l"],
    ]

    for i, target in tqdm(enumerate(train_split)):
        assert target.endswith(
            ".json"
        ), f"Expected target to end with .json, got <{target}>"
        target = target.replace(".json", "")
        metadata = [m for m in metadatas if m["img"].startswith(target)]
        assert (
            len(metadata) == 1
        ), f"Expected one metadata entry for {target}, found {len(metadata)}"
        metadata = metadata[0]
        name_img = metadata["img"].replace(".tif", "")
        # Open the altitude map
        altitude_path = os.path.join(slanted_altitude_dir, f"{name_img}_{ref_key}.iio")
        assert os.path.exists(
            altitude_path
        ), f"Altitude path {altitude_path} does not exist"
        altitude_img = iio.read(altitude_path).squeeze()

        # Create the RangeImageEOGS object
        range_image = RangeImageEOGS(
            metadata=metadata,
            altitude_img=altitude_img,
        )

        # Run TSDF integration
        tsdf_vol.integrate(range_image)

    # Apply prior to the TSDF volume
    tsdf_vol.apply_prior()

    # Extract DSM from TSDF volume and save it
    dsm = tsdf_vol.extract_dsm(
        scene_params=scene_params,
        resolution=(0.3 if "IARPA" in scene_name else 0.5),
        output_dir=output_dir,
    )
    # Export the mesh if necessary
    if export_mesh:
        os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
        print("exporting the mesh to ", output_mesh_path)
        tsdf_vol.extract_mesh(output_mesh_path=output_mesh_path)

    # copy the cfg_dsm
    cfg_rendering = OmegaConf.create(
        OmegaConf.to_container(cfg_rendering, resolve=False)
    )
    cfg_dsm = cfg_rendering.get("eval", None)
    cfg_dsm.pred_dsm_path = os.path.join(output_dir, "dsm.iio")
    cfg_dsm.out_dir = output_dir
    cfg_rendering.iteration = iteration
    cfg_dsm.debug = True  # workaround to remove clearml
    cfg_dsm.prefix = "tsdf:"
    print(" we start to compute the dsm again with the tsdf dsm ")
    mae_tsdf = main_hydra_dsm(cfg_rendering)
    cfg_dsm.prefix = "tsdf_notree:"
    print(" we start to compute the dsm again with the tsdf dsm with no tree  ")
    cfg_dsm.filter_tree = True
    mae_notree_tsdf = main_hydra_dsm(cfg_rendering)
    return mae_tsdf, mae_notree_tsdf


import hydra


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def get_slanted_altitude(cfg_tsdf, cfg_rendering):
    """Get the half_dsm and out dir from the cfg_dsm and cfg.

    The subtelty is if the iteration in the rendering config is -1
    """
    num_iterations = cfg_rendering.get("numiterations", None)
    model_path = cfg_tsdf.get("model_path", None)
    iteration = cfg_rendering.get("iteration", None)
    if iteration != -1:
        return (
            cfg_rendering.slanted_altitude_dir,
            cfg_rendering.out_dir,
            iteration,
            cfg_rendering.output_mesh_path,
        )
    else:
        max_itr = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        if cfg_rendering.optimization.flowmatching.apply_flowmatching:
            name_altitude = f"flowmatched_altitude"
        else:
            name_altitude = f"altitude"
        slanted_altitude_dir = os.path.join(
            model_path, "train_opNone", f"ours_{max_itr}", name_altitude
        )
        output_dir = os.path.join(model_path, "test_opNone", f"ours_{max_itr}", "tsdf")
        output_mesh_path = os.path.join(
            model_path,
            "test_opNone",
            f"ours_{max_itr}",
            "output_mesh",
            "output_mesh.obj",
        )
        return slanted_altitude_dir, output_dir, max_itr, output_mesh_path


@hydra.main(version_base="1.2", config_path="gs_config", config_name="rendering.yaml")
def main_hydra_tsdf(cfg: "Dictconfig"):
    tsdf_cfg = cfg.tsdf
    cfg_rendering = cfg
    debug = cfg.debug
    resume_clearml = tsdf_cfg.resume_clearml
    task_name = tsdf_cfg.task_name
    export_mesh: bool = tsdf_cfg.export_mesh


    if not debug and CLEARML_FOUND:
        print("resume clearml is?", resume_clearml)
        if resume_clearml:
            task = safe_resume_clearml(project_name="EOGS", task_name=task_name)
        else:
            task = safe_init_clearml(project_name="EOGS", task_name=task_name)
    if not cfg.run_tsdf:
        print("you deactivated the TSDF computation, we stop here ")
        return
    slanted_altitude_dir, output_dir, iteration, output_mesh_path = (
        get_slanted_altitude(cfg_rendering=cfg_rendering, cfg_tsdf=tsdf_cfg)
    )
    mae_tsdf, mae_notreetsdf = main(
        slanted_altitude_dir=slanted_altitude_dir,
        scene_name=tsdf_cfg.scene_name,
        vox_size=tsdf_cfg.vox_size,
        trunc_margin_fact=tsdf_cfg.trunc_margin_fact,
        output_dir=output_dir,
        affine_models_json_path=tsdf_cfg.affine_models_json_path,
        train_txt_split_path=tsdf_cfg.train_txt_split_path,
        gt_dir=tsdf_cfg.gt_dir,
        ref_key=tsdf_cfg.ref_key,
        cfg_rendering=cfg_rendering,
        iteration=iteration,
        export_mesh=export_mesh,
        output_mesh_path=output_mesh_path,
    )
    if not debug and CLEARML_FOUND:
        task.get_logger().report_scalar("MAE", "mae_tsdf", value=mae_tsdf, iteration=0)
        task.get_logger().report_scalar(
            "MAE", "mae_notree_tsdf", value=mae_notreetsdf, iteration=0
        )
        task.close()


if __name__ == "__main__":
    # import argparse

    # args = argparse.ArgumentParser()
    # args.add_argument("--slanted_altitude_dir", type=str, default="output/test_1757673654_JAX_068_NEW__rep01/train_opNone/ours_5000/altitude")
    # args.add_argument("--scene_name", type=str, default="JAX_068")
    # args.add_argument("--vox_size", type=float, default=0.5, help="Voxel size in meters")
    # args.add_argument("--trunc_margin_fact", type=float, default=2.5, help="Truncation margin factor for TSDF. The truncation margin is vox_size * trunc_margin_fact")
    # args.add_argument("--output_dir", type=str, required=True, help="Output dir for the extracted DSM and mesh")
    # args.add_argument("--gt_dir", type=str, help="Directory containing ground truth DSMs")
    # args = args.parse_args()
    # main(
    #     slanted_altitude_dir=args.slanted_altitude_dir,
    #     scene_name=args.scene_name,
    #     vox_size=args.vox_size,
    #     trunc_margin_fact=args.trunc_margin_fact,
    #     output_dir=args.output_dir,
    #     affine_models_json_path=os.path.join("data/affine_models",args.scene_name, "affine_models.json"),
    #     train_txt_split_path=os.path.join("data/affine_models",args.scene_name, "train.txt"),
    #     gt_dir=args.gt_dir
    # )
    main_hydra_tsdf()
