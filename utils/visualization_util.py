import igl
from datasets.shape_dataset import *
from utils.geometry_util import torch2np, laplacian_decomposition
from utils.texture_util import generate_tex_coords
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import time
from os import path as osp
def compute_kabsch_rotation(X, Y):
    """
    Compute the optimal rotation matrix R using the Kabsch algorithm
    that minimizes the RMSD between the corresponding points in Y and X.
    In other words, it computes R such that X \approx R @ Y (ignoring translations).

    Args:
        X (np.ndarray): Target points, shape (N, 3).
        Y (np.ndarray): Source points (to be rotated), shape (N, 3).
    
    Returns:
        R (np.ndarray): A 3x3 rotation matrix aligning Y to X.
    """
    # Ensure the arrays are of type float
    X = X.astype(float)
    Y = Y.astype(float)

    # Compute centroids for each set.
    centroid_X = X.mean(axis=0)
    centroid_Y = Y.mean(axis=0)

    # Center the points.
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # Compute the covariance matrix.
    H = Y_centered.T @ X_centered  # (3,3) matrix

    # SVD of the covariance matrix.
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix.
    R = Vt.T @ U.T

    # Correct for a reflection (if necessary)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R

def harmonic_interpolation(V, F, boundary_indices, boundary_values):
    L = igl.cotmatrix(V, F)
    n = V.shape[0]
    interior_indices = np.setdiff1d(np.arange(n), boundary_indices)
    A = L[interior_indices][:, interior_indices]
    b = -L[interior_indices][:, boundary_indices] @ boundary_values
    u_interior = scipy.sparse.linalg.spsolve(A, b)
    u = np.zeros((n, boundary_values.shape[1]))
    u[boundary_indices] = boundary_values
    u[interior_indices] = u_interior
    return u


def get_orientation_calibration_matrix(up_vector, front_vector):
    # align right, up, front dir of the input shape with/into x, y, z axis
    right_vector = np.cross(up_vector, front_vector)
    assert not np.allclose(right_vector, 0)  # ensure no degenerate input
    matrix = np.column_stack((right_vector, up_vector, front_vector)).astype(np.float32)
    return matrix


def orientation_calibration_by_dataset(test_set):
    # Note: While PEP8 suggests line length < 79, keeping these lines unbroken improves readability
    # since the parameters form logical units. This is a case where we can be flexible with PEP8.
    if type(test_set) == PairFaustDataset:  # y up, z front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, 1, 0], front_vector=[0, 0, 1]) 
    elif type(test_set) == PairSmalDataset:  # neg y up, z front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, -1, 0], front_vector=[0, 0, 1]) 
    elif type(test_set) == PairDT4DDataset:  # z up, neg y front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, 0, 1], front_vector=[0, -1, 0]) 
    elif type(test_set) == PairTopKidsDataset:  # z up, neg y front
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, 0, 1], front_vector=[0, -1, 0]) 
    elif type(test_set) == PairCP2PDataset:
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, 0, 1], front_vector=[0, -1, 0]) 
    else:
        print("Unimplemented dataset type, use default orientation matrix y up, z front")
        orientation_matrix = get_orientation_calibration_matrix(up_vector=[0, 1, 0], front_vector=[0, 0, 1])
    return orientation_matrix


def limbs_indices_by_dataset(i, data_x, test_set):
    """Note: Only possible have hand coded for full-full datasets. Maybe there is a better way for partial-partial datasets"""
    if type(test_set) == PairFaustDataset:
        landmarks = np.array([4962, 1249, 77, 2523, 2185])
        limbs_indices = torch2np(data_x["corr"])[landmarks]
    elif type(test_set) == PairSmalDataset:
        landmarks = np.array([1198, 2324, 2523, 1939, 30, 28])  # [hands, feet, head, tail]
        limbs_indices = torch2np(data_x["corr"])[landmarks]
    elif type(test_set) == PairDT4DDataset:
        landmarks = np.array([4819, 2932, 3003, 4091, 3843])  # [hands, feet, head, tail]
        first_index, _ = test_set.combinations[i]
        first_cat = test_set.dataset.off_files[first_index].split('/')[-2]
        if first_cat == 'crypto':
            corr = torch2np(data_x["corr"])
        elif first_cat == 'mousey':
            landmarks = np.array([2583, 6035, 3028, 6880, 78])
            corr = torch2np(data_x["corr"])
        elif first_cat == 'ortiz':
            landmarks = np.array([3451, 7721, 161, 1424, 2021])
            corr = torch2np(data_x["corr"])
        else:
            # Note: Long path construction is more readable on one line
            corr_inter = np.loadtxt(os.path.join(test_set.dataset.data_root, 'corres', 'cross_category_corres', f'crypto_{first_cat}.vts'), dtype=np.int32) - 1
            corr_intra = torch2np(data_x["corr"])
            corr = corr_intra[corr_inter]
        limbs_indices = corr[landmarks]
    elif type(test_set) == PairTopKidsDataset:
        limbs_indices = np.array([8438, 7998, 11090, 11416, 9885])  # [hands, feet, head
    else:
        print("Unimplemented dataset type, colored limb indices(hands, feets, head, tail) are not defined.")
        limbs_indices = np.arange(6)
    return limbs_indices


# Note: Function signature split for better readability while maintaining reasonable line length
def compute_partial_texture_mapping(verts_x, verts_y, faces_x, faces_y, evecs_x, evecs_trans_x,
                                  ours_corr_x, ours_corr_y, ours_overlap_score12, ours_overlap_score21):
    """
    Compute texture mapping between two meshes using spectral Laplacian decomposition on the overlapping region.

    This function constructs a smooth mapping from texture coordinates on mesh X to the overlapping
    region of mesh Y. It performs the following steps:
    1. Builds a correspondence matrix (P) using given per-vertex correspondences.
    2. Converts per-vertex overlap scores to per-face overlap masks for both meshes.
    3. Extracts the submesh of mesh Y that lies within the overlapping region.
    4. Computes a localized Laplacian decomposition over the submesh.
    5. Uses the spectral bases to compute a smooth transfer operator (Pyx) from mesh X to mesh Y.
    6. Generates texture coordinates for mesh X and maps them to mesh Y.
    7. Constructs "corner" UV coordinates for each face and masks out non-overlapping areas.
    Note:
        The current implementation uses a submesh extraction method to isolate the overlapping region for
        spectral processing, this is sensitive to the number of eigenvectors used in evecs_x and evecs_sub.
        While effective, there may exist alternative methods that could improve
        robustness and efficiency in partial-to-partial texture transfers.
    """
    # Build correspondence matrix
    P = np.zeros((len(verts_y), len(verts_x)))
    P[ours_corr_y, ours_corr_x] = 1

    # Process overlap masks from per vertex to per face
    face_mask_x = np.all(ours_overlap_score12[faces_x], axis=1).astype(np.float32)
    face_mask_y = np.all(ours_overlap_score21[faces_y], axis=1).astype(np.float32)

    # Extract submesh of overlapping region
    selected_faces = faces_y[face_mask_y == 1]
    unique_vertices, inverse_indices = np.unique(selected_faces.flatten(), return_inverse=True)
    sub_faces = inverse_indices.reshape(selected_faces.shape)
    sub_vertices = verts_y[unique_vertices]

    P = P[unique_vertices]

    # smooth Pyx from x to submesh_y
    _, evecs_sub, evecs_trans_sub, _ = laplacian_decomposition(sub_vertices, sub_faces, k=80)
    Cxy = evecs_trans_sub @ P @ evecs_x
    Pyx = evecs_sub @ Cxy @ evecs_trans_x

    # Generate and map texture coordinates
    tex_coords_x = generate_tex_coords(verts_x)
    tex_coords_y_sub = Pyx @ tex_coords_x
    tex_coords_y = np.zeros((len(verts_y), 2))
    tex_coords_y[unique_vertices] = tex_coords_y_sub

    # Create corner UVs
    corner_uv_x = tex_coords_x[faces_x]
    corner_uv_y = tex_coords_y[faces_y]
    corner_uv_x[face_mask_x == 0] = [0, 0]
    corner_uv_y[face_mask_y == 0] = [0, 0]

    return corner_uv_x, corner_uv_y

def get_rotation_matrix(rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]]).T
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]]).T
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]]).T
    return Rz @ Ry @ Rx

class VisualizationManager:
    def __init__(self, results_root):
        # Initialize visualization state
        # Separate rotation controls for first and second shapes
        self.rot_x1 = self.rot_y1 = 30  # First shape rotation
        self.rot_z1 = 0
        self.rot_x2 = self.rot_y2 = 30  # Second shape rotation
        self.rot_z2 = 0
        
        # Layout controls
        self.column_gap = [0.8, 0, 0]  # Gap between columns
        self.row_gap = [0, -0.8, 0]    # Gap between pairs in each row
        self.layout_mode = "column"     # Toggle between "column" and "row"
        
        # UV controls
        self.uv_scale = 1.0
        self.uv_pan_u = self.uv_pan_v = 0.0
        
        self.verts_x = None
        self.verts_y = None
        self.results_root = results_root
        self.registered_methods = []  # Track registered visualization methods
        self.texture_callback = None  # Callback for texture recomputation
    
    def get_layout_offset(self, idx, is_second_shape=False):
        """Compute offset based on current layout mode"""
        if self.layout_mode == "column":
            offset = np.array(self.column_gap) * idx
            if is_second_shape:
                offset = offset + self.row_gap
        else:  # row mode
            offset = np.array(self.row_gap) * idx
            if is_second_shape:
                offset = offset + self.column_gap
        return offset

    def register_meshes(self, verts_x, verts_y, faces_x, faces_y, method_prefixes):
        """
        Register meshes for multiple correspondence methods
        Args:
            verts_x, verts_y: Vertex positions for both shapes
            faces_x, faces_y: Face indices for both shapes
            method_prefixes: List of method names to register (e.g. ["gt", "ours", "baseline1"])
        """
        self.verts_x = verts_x
        self.verts_y = verts_y
        self.registered_methods = method_prefixes
        
        R1 = get_rotation_matrix(self.rot_x1, self.rot_y1, self.rot_z1)
        R2 = get_rotation_matrix(self.rot_x2, self.rot_y2, self.rot_z2)
        verts_x = verts_x @ R1
        verts_y = verts_y @ R2
        
        # Register meshes for each method
        for idx, prefix in enumerate(method_prefixes):
            offset_first = self.get_layout_offset(idx)
            offset_second = self.get_layout_offset(idx, is_second_shape=True)
            ps.register_surface_mesh(f"{prefix}_first", verts_x + offset_first, faces_x, material="wax", smooth_shade=True)
            ps.register_surface_mesh(f"{prefix}_second", verts_y + offset_second, faces_y, material="wax", smooth_shade=True)

    def update_mesh_positions(self):
        """Update mesh positions based on current state"""
        R1 = get_rotation_matrix(self.rot_x1, self.rot_y1, self.rot_z1)
        R2 = get_rotation_matrix(self.rot_x2, self.rot_y2, self.rot_z2)
        
        for prefix in self.registered_methods:
            idx = self.registered_methods.index(prefix)
            offset_first = self.get_layout_offset(idx)
            offset_second = self.get_layout_offset(idx, is_second_shape=True)
            
            # Update first shape with R1
            mesh_first = ps.get_surface_mesh(f"{prefix}_first")
            pos_first = self.verts_x @ R1 + offset_first
            mesh_first.update_vertex_positions(pos_first)
            
            # Update second shape with R2
            mesh_second = ps.get_surface_mesh(f"{prefix}_second")
            pos_second = self.verts_y @ R2 + offset_second
            mesh_second.update_vertex_positions(pos_second)

    def add_texture_visualization(self, prefix, corner_uv_x, corner_uv_y, texture):
        """Add texture parameterization for correspondence visualization"""
        # Apply UV transformation
        corner_uv_x = corner_uv_x * self.uv_scale + np.array([self.uv_pan_u, self.uv_pan_v])
        corner_uv_y = corner_uv_y * self.uv_scale + np.array([self.uv_pan_u, self.uv_pan_v])
        
        ps.get_surface_mesh(f"{prefix}_first").add_parameterization_quantity("para", corner_uv_x.reshape(-1, 2), defined_on='corners', enabled=True)
        ps.get_surface_mesh(f"{prefix}_second").add_parameterization_quantity("para", corner_uv_y.reshape(-1, 2), defined_on='corners', enabled=True)
        ps.get_surface_mesh(f"{prefix}_first").add_color_quantity("texture", texture, defined_on='texture', param_name="para", enabled=True)
        ps.get_surface_mesh(f"{prefix}_second").add_color_quantity("texture", texture, defined_on='texture', param_name="para", enabled=True)

    def callback(self):
        """Polyscope callback for interactive controls"""
        # Rotation controls for first shape
        psim.Text("Rotate first shape:")
        changed_x1, self.rot_x1 = psim.SliderFloat("rot_x1", self.rot_x1, v_min=-180, v_max=180)
        changed_y1, self.rot_y1 = psim.SliderFloat("rot_y1", self.rot_y1, v_min=-180, v_max=180)
        changed_z1, self.rot_z1 = psim.SliderFloat("rot_z1", self.rot_z1, v_min=-180, v_max=180)
        
        # Rotation controls for second shape
        psim.Text("Rotate second shape:")
        changed_x2, self.rot_x2 = psim.SliderFloat("rot_x2", self.rot_x2, v_min=-180, v_max=180)
        changed_y2, self.rot_y2 = psim.SliderFloat("rot_y2", self.rot_y2, v_min=-180, v_max=180)
        changed_z2, self.rot_z2 = psim.SliderFloat("rot_z2", self.rot_z2, v_min=-180, v_max=180)
        
        # Layout controls
        psim.Text("Layout:")
        changed_col_gap, self.column_gap[0] = psim.SliderFloat("column_gap", self.column_gap[0], v_min=0.5, v_max=2.0)
        changed_row_gap, self.row_gap[1] = psim.SliderFloat("row_gap", self.row_gap[1], v_min=-2.0, v_max=0.0)
        
        # UV controls
        psim.Text("Texture controls:")
        changed_scale, self.uv_scale = psim.SliderFloat("scale", self.uv_scale, v_min=0.1, v_max=2.0)
        changed_u, self.uv_pan_u = psim.SliderFloat("pan_u", self.uv_pan_u, v_min=-1.0, v_max=1.0)
        changed_v, self.uv_pan_v = psim.SliderFloat("pan_v", self.uv_pan_v, v_min=-1.0, v_max=1.0)
        
        # Buttons
        if psim.Button("Reset View"):
            self.rot_x1 = self.rot_y1 = self.rot_x2 = self.rot_y2 = 30
            self.rot_z1 = self.rot_z2 = 0
            self.column_gap[0] = 0.8
            self.row_gap[1] = -0.8
            self.uv_scale = 1.0
            self.uv_pan_u = self.uv_pan_v = 0.0
            changed_x1 = True
        
        if psim.Button("Toggle Layout"):
            self.layout_mode = "row" if self.layout_mode == "column" else "column"
            self.update_mesh_positions()
        
        if psim.Button("Save Screenshot"):
            ps.screenshot(osp.join(self.results_root, "polyscope_visualization", f"view_{time.strftime('%Y%m%d_%H%M%S')}.png"))

        if psim.Button("Re-Texture") and self.texture_callback is not None:
            self.texture_callback(get_rotation_matrix(self.rot_x1, self.rot_y1, self.rot_z1),
                                get_rotation_matrix(self.rot_x2, self.rot_y2, self.rot_z2))
        
        # Update positions if needed
        if any([changed_x1, changed_y1, changed_z1, 
                changed_x2, changed_y2, changed_z2,
                changed_col_gap, changed_row_gap]):
            self.update_mesh_positions()
            
        # DONT Update textures if UV controls changed, slow, think about a better implementation in the future
        # if any([changed_scale, changed_u, changed_v]) and self.texture_callback is not None:
        #     self.texture_callback(get_rotation_matrix(self.rot_x1, self.rot_y1, self.rot_z1),
        #                         get_rotation_matrix(self.rot_x2, self.rot_y2, self.rot_z2))