import scipy.sparse.linalg as sla
from utils.geometry_util import hash_arrays, torch2np


import numpy as np
import scipy.sparse
import torch


import os
import os.path as osp
import igl
# import pyshell


def computeEV(v, f, k, bending_weight=1e-2, sigma=None, which=None, nonzero=True):
    """
    Compute eigenvectors of shell hessian 
        Args:
            fix (optional): vertex indices for boundary values

        returns:
            vals : eigenvalues
            vecs: eigenvectors 3n x k 
    """

    f = f.astype(np.int32)
    # edge flaps
    uE, EMAP, EF, EI = igl.edge_flaps(f)
    # massmatrix
    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    M = scipy.sparse.block_diag((M, M, M), format='lil')
    k = k + 6

    hess = pyshell.shell_deformed_hessian(v, v, f, uE, EMAP, EF, EI, bending_weight)

    # hessian is quite poorly conditioned
    eps = 1e-6
    hess = (hess + eps * scipy.sparse.identity(hess.shape[0])).tocsc()

    if sigma is None:
        if which is None:
            # note: sigma is no longer equal to 0 like in original implementation
            sigma = eps
            which = 'LM'
            vals, vecs = sla.eigsh(hess, k, M=M, sigma=0, which=which)
            # vals_eps, vecs_eps = sla.eigsh(hess_eps, k, M=M, sigma=sigma, which=which)
        else:
            vals, vecs = sla.eigsh(hess, k, M=M, which=which)
    else:
        vals, vecs = sla.eigsh(hess, k, M=M, sigma=sigma)

    if nonzero:  # remove vecs corresponding to zero vals (six dimensional kernel due to rigid body motions)
        ind = vals > 1e-8

        if np.sum(ind) < k - 6:
            print('mesh has probably irregularities, found ' + str(k - 6 - np.sum(ind)) + 'many EF with zero EV')
            # vals,vecs =  linalg.eigsh(hess, k+np.sum(ind)-6,M=M,sigma = sigma,which = which)
            ind = vals > 1e-8

        vals = vals[6:]
        vecs = vecs[:, 6:]

    # fix signs
    ind = vecs[0, :] < 0
    vals[ind] *= -1
    vecs[:, ind] *= -1

    return vals, vecs


def computeVectorBasis(v, f, k=200, bending_weight=1e-2, nonzero=True):
    """
        computes elastic vibration modes (i.e. vector valued eigenfunctions of elastic hessian)
        args:
            k: first k eigennfunctions (lowest eigenvalues)
            bending_weight: weighting term of membrane and bending energy 
            nonzero: (boolean) compute eigenfunctions with nonzero eigenvalue (kernel with r.b.m. is removed)
    """
    # print("Computing " + str(k) + ' elastic eigenfunctions for ' + self.name)
    Evalues, vectorBasis = computeEV(v, f, k, bending_weight=bending_weight,
                                                        nonzero=nonzero)
    return Evalues, vectorBasis


def projection(normals):
    """
    Create a projection matrix for projection on vertex normals
        args:
            normals: m x 3

        returns:
            P: projection matrix 3*m x m 

    """
    # normals in m x 3
    m = normals.shape[0]
    P = scipy.sparse.lil_matrix((3 * m, m), dtype='d')
    for j in range(m):
        P[j, j] = normals[j, 0]
        P[j + m, j] = normals[j, 1]
        P[j + 2 * m, j] = normals[j, 2]
    return P


def computeElasticBasis(v, f, k=200, bending_weight=1e-2, nonzero=True, rescale_unit_area=False):

    if rescale_unit_area:
            area = np.sqrt(np.sum(igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)))
            v = (v / area)
    f = f.astype(np.int32)
    nVert = v.shape[0]
    mass = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    normals = igl.per_vertex_normals(v, f)
    adj = igl.adjacency_list(f)

    Evalues, vectorBasis = computeVectorBasis(v, f, k=k, bending_weight=bending_weight, nonzero=nonzero)
    elasticBasis = projection(normals).T.dot(vectorBasis)

    return mass, Evalues, elasticBasis


def get_elas_operators(verts, faces, k=200, bending_weight=1e-2, normals=None,
                  cache_dir=None, overwrite_cache=False):
    """
    See documentation for get_operators(). This is just its parrallel implementation for elastic case.
    This essentailly just wraps a call to elastic basis computation, using a cache if possible.
    All arrays are always computed using double precision for stability,
    then truncated to single precision floats to store on disk,
    and finally returned as a tensor with dtype/device matching the `verts` input.
    """
    assert verts.dim() == 2, 'Please call get_all_operators() for a batch of vertices'
    device = verts.device
    dtype = verts.dtype
    verts_np = torch2np(verts.float()) # use 32 in hash search and in production
    faces_np = torch2np(faces) if faces is not None else None

    if np.isnan(verts_np).any():
        raise ValueError('detect NaN vertices.')

    found = False
    if cache_dir:
        assert osp.isdir(cache_dir)
        hash_key_str = str(bending_weight) + str(hash_arrays((verts_np, faces_np)))

        # Search through buckets with matching hashes.
        # When the loop exits,
        # this is the bucket index of the file we should write to.
        i_cache = 0
        while True:
            # From the name of the file to check
            search_path = osp.join(cache_dir, hash_key_str+'_'+str(i_cache)+'.npz')

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile['verts']
                cache_faces = npzfile['faces']
                cache_k = npzfile['k_eig'].item()

                # If the cache doesn't match, keep searching
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache += 1
                    print('collision detected')
                    continue

                # Delete previous file and overwrite it
                if overwrite_cache or cache_k < k:
                    os.remove(search_path)
                    break

                def read_sp_mat(prefix):
                    data = npzfile[prefix + '_data']
                    indices = npzfile[prefix + '_indices']
                    indptr = npzfile[prefix + '_indptr']
                    shape = npzfile[prefix + '_shape']
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # this entry matches. return it.
                mass = npzfile['mass']
                evals = npzfile['evals'][:k]
                evecs = npzfile['evecs'][:, :k]

                mass = mass.item().diagonal()
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)

                found = True
                break
            except FileNotFoundError:
                # not found, create a new file
                break
            except Exception:
                # any other exception might indicate bad zip file, delete it and overwrite it
                os.remove(search_path)
                break

    if not found:
        # recompute
        mass, evals, evecs = computeElasticBasis(torch2np(verts), faces_np, k=200, bending_weight=bending_weight) # use double precision in precomputation for stability

        dtype_np = np.float32

        # save
        if cache_dir:
            mass_np = mass.astype(dtype_np)
            evals_np = evals.astype(dtype_np)
            evecs_np = evecs.astype(dtype_np)


            np.savez(
                search_path,
                verts=verts_np,
                faces=faces_np,
                k_eig=k,
                mass=mass_np,
                evals=evals_np,
                evecs=evecs_np,
            )
            mass = mass.diagonal()
            mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
            evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
            evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)

    return mass, evals, evecs


def projectOnNormals(v, f, function):
    """
    project a 3n function on vertex normals
        args:
            v, f: mesh
            function: n x 3
        returns:
            proj : scalar valued function on mesh
    """

    normals = igl.per_vertex_normals(v, f)

    proj = [normals[i] * np.dot(function[i], normals[i]) for i in range(normals.shape[0])]

    return proj