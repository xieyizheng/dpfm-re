import os
import re
import torch
import scipy.linalg
import numpy as np
from utils.geometry_util import get_operators, get_elas_operators, get_geodesic_distmat

def get_shape_operators_and_data(item, cache_dir, config):
    """Get spectral and elastic operators for a shape."""
    verts, faces = item['verts'], item['faces']

    if config.get('return_evecs', True):
        item = get_spectral_ops(
            item,
            num_evecs=config.get('num_evecs', 200),
            cache_dir=os.path.join(cache_dir, 'diffusion')
        )

    if config.get('return_elas_evecs', False):
        item = get_elas_spectral_ops(
            item,
            num_evecs=config.get('num_evecs', 200),
            bending_weight=config.get('bending_weight', 1e-2),
            cache_dir=os.path.join(cache_dir, 'elastic')
        )
    
    if config.get('return_dist', False):
        item['dist'] = get_geodesic_distmat(verts, faces, cache_dir=os.path.join(cache_dir, 'dist'))
    
    # xyz
    item['xyz'] = verts
    
    return item

def get_spectral_ops(item, num_evecs, cache_dir=None, dirichlet=False):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    
    # Use max of num_evecs and 128 for get_operators
    k = max(num_evecs, 128)
    _, mass, L, evals, evecs, gradX, gradY = get_operators(item['verts'], item.get('faces'),
                                                   k=k,
                                                   cache_dir=cache_dir)
    
    # Store 128-length operators in diffusion dict
    item['operators'] = {
        'evecs': evecs[:, :128],
        'evecs_trans': (evecs.T * mass[None])[:128],
        'evals': evals[:128],
        'mass': mass,
        'L': L,
        'gradX': gradX, 
        'gradY': gradY
    }
    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    item['L'] = L

    return item

def get_elas_spectral_ops(item, num_evecs, bending_weight=1e-2, cache_dir=None):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    mass, evals, evecs = get_elas_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs, bending_weight=bending_weight, 
                                                   cache_dir=cache_dir)
    evecs = treat_nan(evecs, item['verts'], item['name'])
    mass = mass + 1e-8 * mass.mean()
    sqrtmass = torch.sqrt(mass)
    
    def const_proj(evec, sqrtmass):
        # orthogonal projector for elastic basis
        sqrtM = torch.diag(sqrtmass)
        return torch.linalg.pinv(sqrtM @ evec) @ sqrtM
    evecs_trans = const_proj(evecs[:, :num_evecs], sqrtmass)
    Mk = evecs.T @ torch.diag(mass) @ evecs
    item['elas_evecs'] = evecs[:, :num_evecs]
    item['elas_evecs_trans'] = evecs_trans[:num_evecs]
    item['elas_evals'] = evals[:num_evecs]
    item['elas_mass'] = mass
    item['elas_Mk'] = Mk
    sqrtMk = scipy.linalg.sqrtm(to_numpy(Mk)).real #numerical weirdness 
    sqrtMk = torch.tensor(sqrtMk).float().to(Mk.device)
    invsqrtMk = torch.linalg.pinv(sqrtMk)
    item['elas_sqrtMk'] = sqrtMk
    item['elas_invsqrtMk'] = invsqrtMk
    return item














def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)

def treat_nan(tensor, verts, name):
    """
    Replaces NaN values in a tensor with the nearest non-NaN values from a given vertices tensor.

    Args:
        tensor (torch.Tensor): The input tensor containing potential NaN values.
        verts (torch.Tensor): The vertices tensor used for finding nearest non-NaN entries.
        name (str): The name of the tensor for informative output.

    Returns:
        torch.Tensor: The modified tensor with NaN values replaced.
    """
    # Detect rows with NaN values
    nan_rows = torch.isnan(tensor).any(dim=1)
    if nan_rows.any():
        num_nan_rows = nan_rows.sum().item()
        nan_row_indices = nan_rows.nonzero(as_tuple=True)[0]
        print(f'Warning: {num_nan_rows} rows have NaN values in {name}, replacing with nearest non-NaN neighbor')
        print(f'Row numbers: {nan_row_indices}')

    # Identify non-NaN rows
    non_nan_rows = ~nan_rows
    # Calculate distances to nearest non-NaN neighbors
    nan_verts = verts[nan_rows]
    non_nan_verts = verts[non_nan_rows]
    distances = torch.cdist(nan_verts, non_nan_verts)
    _, nearest_indices = torch.min(distances, dim=1)

    # Replace NaN rows with their nearest non-NaN neighbors
    tensor[nan_rows] = tensor[non_nan_rows][nearest_indices]

    return tensor