# import dino
from utils.geometry_util import hash_arrays, torch2np
import numpy as np
import torch
import os
import os.path as osp
from utils.cache_util import get_cached_compute

def compute_dino_features(verts, faces, patch_size=14, model_type='dino_vits8'):
    """
    Compute DINO features for a mesh.
    
    Args:
        verts (np.ndarray): Vertex positions [V, 3]
        faces (np.ndarray): Face indices [F, 3]
        patch_size (int): Patch size for DINO model
        model_type (str): DINO model type to use
        
    Returns:
        np.ndarray: DINO features
    """
    return dino.get_features(
        verts.astype(np.float64),
        faces,
        patch_size=patch_size,
        model_type=model_type
    )

def get_dino_features(verts, faces, cache_dir=None, overwrite_cache=False):
    """
    Get DINO features for a mesh, using caching if possible.
    
    Args:
        verts (torch.Tensor): Vertex positions [V, 3]
        faces (torch.Tensor): Face indices [F, 3]
        cache_dir (str, optional): Directory to cache results. Default None.
        overwrite_cache (bool): Whether to overwrite existing cache. Default False.
        
    Returns:
        torch.Tensor: DINO features
    """
    return get_cached_compute(compute_dino_features, verts, faces, cache_dir=cache_dir,
                            patch_size=14, model_type='dino_vits8')
