import numpy as np
import torch
from utils.tensor_util import to_numpy
import scipy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def spectral_mass_computation(elas_evecs, mass):
    M = torch.diag(mass)
    Mk = elas_evecs.t() @ M @ elas_evecs
    sqrtMk = scipy.linalg.sqrtm(to_numpy(Mk)).real #numerical weirdness
    sqrtMk = torch.tensor(sqrtMk).to(elas_evecs.device).float()
    invsqrtMk = torch.linalg.pinv(sqrtMk)
    return Mk, sqrtMk, invsqrtMk

def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    p2p = dist.argmin(dim=dim)
    return p2p

def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Compute functional map from point-to-point map
    Arg:
        p2p: point-to-point map (shape y -> shape x). [V2]
    Return:
        Cxy: functional map (shape x -> shape y). Shape [K, K]
    """
    evecs_x_a = evecs_x[p2p]
    evecs_y_a = evecs_y

    Cxy = torch.linalg.lstsq(evecs_y_a, evecs_x_a).solution
    return Cxy

def elas_pointmap2fmap(p2p, evecs_x, evecs_y, mass_x, mass_y):
    """
    Compute general(elastic) functional map from point-to-point map
    Args:
        p2p: point-to-point map (shape y -> shape x). [V2]
    Returns:
        Cxy: functional map (shape x -> shape y). Shape [K, K]
    """
    evecs_x_a = evecs_x[p2p]
    evecs_y_a = evecs_y
    mass_y_a = mass_y
    sqrt_mass_y_a = torch.sqrt(mass_y_a)
    evecs_x_a = torch.diag(sqrt_mass_y_a) @ evecs_x_a
    evecs_y_a = torch.diag(sqrt_mass_y_a) @ evecs_y_a

    Cxy = torch.linalg.lstsq(evecs_y_a, evecs_x_a).solution
    return Cxy

def _fmap2pointmap(Cxy, evecs_x, evecs_y):
    """
    helper function to convert functional map to point-to-point map
    """
    dataA = evecs_x @ Cxy.t()
    dataB = evecs_y

    return dataA, dataB

def fmap2pointmap(Cxy, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        Cxy: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    dataA, dataB = _fmap2pointmap(Cxy, evecs_x, evecs_y)
    p2p = nn_query(dataA, dataB)
    return p2p

def _elas_fmap2pointmap(elas_Cxy, elas_evecs_x, elas_evecs_y, mass1, mass2):
    """
    helper function to convert general(elastic) functional map to point-to-point map
    """
    M1k, sqrtM1k, invsqrtM1k = spectral_mass_computation(elas_evecs_x, mass1)
    M2k, sqrtM2k, invsqrtM2k = spectral_mass_computation(elas_evecs_y, mass2)

    dataA = elas_evecs_x @ torch.inverse(M1k) @ elas_Cxy.t() @ M2k @ invsqrtM2k
    dataB = elas_evecs_y @ invsqrtM2k

    return dataA, dataB

def elas_fmap2pointmap(elas_Cxy, elas_evecs_x, elas_evecs_y, mass1, mass2):
    """
    Convert general(elastic) functional map to point-to-point map
    
    Args:
        elas_Cxy: general(elastic) functional map (shape x->shape y). Shape [K, K]
        elas_evecs_x: eigenvectors of shape x. Shape [V1, K]
        elas_evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    dataA, dataB = _elas_fmap2pointmap(elas_Cxy, elas_evecs_x, elas_evecs_y, mass1, mass2)
    return nn_query(dataA, dataB)

def hybrid_fmap2pointmap(Cxy, evecs_x, evecs_y, elas_Cxy, elas_evecs_x, elas_evecs_y, mass1, mass2):
    """
    Convert hybrid functional map to point-to-point map
    
    Args:
        Cxy: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
        elas_Cxy: general(elastic) functional map (shape x->shape y). Shape [K, K]
        elas_evecs_x: eigenvectors of shape x. Shape [V1, K]
        elas_evecs_y: eigenvectors of shape y. Shape [V2, K]
        mass1: mass of shape x. Shape [V1]
        mass2: mass of shape y. Shape [V2]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    lb_dataA, lb_dataB = _fmap2pointmap(Cxy, evecs_x, evecs_y)
    elas_dataA, elas_dataB = _elas_fmap2pointmap(elas_Cxy, elas_evecs_x, elas_evecs_y, mass1, mass2)

    # merge embeddings
    merged_dataA = torch.cat((elas_dataA, lb_dataA), dim=1)
    merged_dataB = torch.cat((elas_dataB, lb_dataB), dim=1)

    p2p = nn_query(merged_dataA, merged_dataB)

    return p2p

def zoomout(p2p, evecs_x, evecs_y, k_init=30, k_final=200):
    assert evecs_x.size(1) >= k_final

    for k in tqdm(range(k_init, k_final+1)):
        Cxy = pointmap2fmap(p2p, evecs_x[:, :k], evecs_y[:, :k])
        p2p = fmap2pointmap(Cxy, evecs_x[:, :k], evecs_y[:, :k])
    
    return p2p

def elas_zoomout(p2p, evecs_x, evecs_y, mass1, mass2, k_init=30, k_final=200):
    assert evecs_x.size(1) >= k_final

    for k in tqdm(range(k_init, k_final+1)):
        Cxy = elas_pointmap2fmap(p2p, evecs_x[:, :k], evecs_y[:, :k], mass1, mass2)
        p2p = elas_fmap2pointmap(Cxy, evecs_x[:, :k], evecs_y[:, :k], mass1, mass2)
    
    return p2p

def hybrid_zoomout(p2p, evecs_x, evecs_y, elas_evecs_x, elas_evecs_y, mass1, mass2, k_init=(20,10), k_final=(100,100)):
    assert evecs_x.size(1) >= k_final[0]
    assert elas_evecs_x.size(1) >= k_final[1]

    start_lb, start_elas = k_init
    end_lb, end_elas = k_final
    steps = sum(k_final) - sum(k_init)
    step_lb = (end_lb - start_lb) / steps
    step_elas = (end_elas - start_elas) / steps

    for k in tqdm(range(steps+1)):
        k_lb = int(start_lb + k * step_lb)
        k_elas = int(start_elas + k * step_elas)

        Cxy = pointmap2fmap(p2p, evecs_x[:, :k_lb], evecs_y[:, :k_lb])
        elas_Cxy = elas_pointmap2fmap(p2p, elas_evecs_x[:, :k_elas], elas_evecs_y[:, :k_elas], mass1, mass2)
        
        p2p = hybrid_fmap2pointmap(Cxy, evecs_x[:, :k_lb], evecs_y[:, :k_lb], elas_Cxy, elas_evecs_x[:, :k_elas], elas_evecs_y[:, :k_elas], mass1, mass2)
    
    return p2p

def trim_basis(data, n, elas_n):
    """
    trim the spectral operators (both LB ad Elastic) to specified numbers, intended for mixing up basis Fmap computation

    """
    # everthing has a batch dim
    evals = data['evals']
    elas_evals = data['elas_evals']
    data['evals'] = data['evals'][:, :n]
    data['evecs'] = data['evecs'][:, :, :n]
    data['evecs_trans'] = data['evecs_trans'][:, :n, :]
    if elas_n > 0:
        data['elas_evals'] = elas_evals[:, :elas_n]
        data['elas_evecs'] = data['elas_evecs'][:, :, :elas_n]

        # recompute the elastic evecs_trans
        mass = data['elas_mass'].squeeze(0)
        sqrtmass = torch.sqrt(mass)
        evecs = data['elas_evecs'].squeeze(0)
        def const_proj(evec, sqrtmass):
            # orthogonal projector for elastic basis
            sqrtM = torch.diag(sqrtmass)
            return torch.linalg.pinv(sqrtM @ evec) @ sqrtM
        evecs_trans = const_proj(evecs, sqrtmass)

        data['elas_evecs_trans'] = evecs_trans.unsqueeze(0)

        # recompute the elastic Mk
        Mk = evecs.T @ torch.diag(mass) @ evecs
        data['elas_Mk'] = Mk.unsqueeze(0)

        # recompute sqrtMk and invsqrtMk
        sqrtMk = scipy.linalg.sqrtm(to_numpy(Mk)).real #numerical weirdness from LB
        sqrtMk = torch.tensor(sqrtMk).float().to(Mk.device)
        invsqrtMk = torch.linalg.pinv(sqrtMk)
        data['elas_sqrtMk'] = sqrtMk.unsqueeze(0)
        data['elas_invsqrtMk'] = invsqrtMk.unsqueeze(0)

    return data


def corr2fmap(corr_x, corr_y, evecs_x, evecs_y):
    """
    Compute functional map from correspondences
    Cxy : shape x -> shape y
    """
    evecs_x_a = evecs_x[corr_x]
    evecs_y_a = evecs_y[corr_y]

    Cxy = torch.linalg.lstsq(evecs_y_a, evecs_x_a).solution
    return Cxy

def elas_corr2fmap(corr_x, corr_y, evecs_x, evecs_y, mass_x, mass_y):
    """
    Compute general(elastic) functional map from correspondences
    Cxy : shape x -> shape y
    """
    evecs_x_a = evecs_x[corr_x]
    evecs_y_a = evecs_y[corr_y]
    mass_y_a = mass_y[corr_y]
    sqrt_mass_y_a = torch.sqrt(mass_y_a)
    evecs_x_a = torch.diag(sqrt_mass_y_a) @ evecs_x_a
    evecs_y_a = torch.diag(sqrt_mass_y_a) @ evecs_y_a

    Cxy = torch.linalg.lstsq(evecs_y_a, evecs_x_a).solution
    return Cxy

def elas_Cfromlandmark(evec1, evec2, mass1, mass2, vts1, vts2):
    evec1_a = evec1[vts1]
    evec2_a = evec2[vts2]

    mass1_a = mass1[vts1]
    mass2_a = mass2[vts2]

    F = evec1_a
    G = evec2_a

    FMF = F.t() @ torch.diag(mass1_a) @ F
    FMG = F.t() @ torch.diag(mass1_a) @ G

    return torch.linalg.lstsq(FMF, FMG)[0]

def dev_elas_Cfromlandmark(evec1, evec2, mass1, mass2, vts1, vts2):
    evec1_a = evec1[vts1]
    evec2_a = evec2[vts2]

    mass1_a = mass1[vts1]
    mass2_a = mass2[vts2]

    F = evec1_a
    G = evec2_a

    M1ka = F.t() @ torch.diag(mass1_a) @ F

    return torch.linalg.inv(M1ka) @ evec1_a.t() @ torch.diag(mass1_a) @ evec2_a
def elas_p2pfromFmap(C, evec1, evec2, mass1, mass2):
    """
    compute p2p that is generalizable to both Elastic C and LB C
    assume the C12 is like Phi1 P Phi2
    """
    M1 = torch.diag(mass1)
    M2 = torch.diag(mass2)

    M2k = evec2.t() @ M2 @ evec2
    sqrtM2k = scipy.linalg.sqrtm(to_numpy(M2k)).real #numerical weirdness from LB
    sqrtM2k = torch.tensor(sqrtM2k).to(C.device)
    invsqrtM2k = torch.linalg.pinv(sqrtM2k)

    M1k = evec1.t() @ M1 @ evec1
    sqrtM1k = scipy.linalg.sqrtm(to_numpy(M1k)).real #numerical weirdness from LB
    sqrtM1k = torch.tensor(sqrtM1k).to(C.device)
    invsqrtM1k = torch.linalg.pinv(sqrtM1k)

    dataA = evec1 @ C.t() @ invsqrtM2k.float()
    dataB = evec2 @ invsqrtM2k.float()

    def const_proj(evec, M):
        M = torch.diag(M)
        return torch.linalg.pinv(evec.t() @ M @ evec) @ evec.t() @ M
    proj1 = const_proj(evec1, mass1)
    proj2 = const_proj(evec2, mass2)
    # #my new NN search objective

    # _, pred_labels2to1 = diffusion_net.geometry.find_knn(dataB, dataA, k=1, method='cpu_kd')
    # pred_labels2to1 = pred_labels2to1.squeeze(-1)
    
    pred_labels2to1 = nn_query(dataA, dataB) 

    return pred_labels2to1

def dev_combined_elas_p2pfromFmap(C, evec1, evec2, mass1, mass2, C_LB, evecx, evecy):
    """
    compute p2p that is generalizable to both Elastic C and LB C
    assume the C12 is like Phi1 P Phi2
    """
    M1 = torch.diag(mass1)
    M2 = torch.diag(mass2)
    # add support for when evec1 is empty or single column
    if evec1.size(1) == 0 or evec1.size(1) == 1:
        # dataA and dataB are set to empty tensor
        dataA = evec1
        dataB = evec2

    else:
        M2k = evec2.t() @ M2 @ evec2
        sqrtM2k = scipy.linalg.sqrtm(to_numpy(M2k)).real #numerical weirdness from LB
        sqrtM2k = torch.tensor(sqrtM2k).to(C.device)
        invsqrtM2k = torch.linalg.pinv(sqrtM2k)

        M1k = evec1.t() @ M1 @ evec1
        sqrtM1k = scipy.linalg.sqrtm(to_numpy(M1k)).real #numerical weirdness from LB
        sqrtM1k = torch.tensor(sqrtM1k).to(C.device)
        invsqrtM1k = torch.linalg.pinv(sqrtM1k)

        dataA = evec1 @ C.t() @ invsqrtM2k.float()
        dataB = evec2 @ invsqrtM2k.float()

    data1 = evecx @ C_LB.t()
    data2 = evecy

    # add support for when one of the data is empty
    if dataA.size(1) == 0:
        dataA = data1
        dataB = data2
    elif data1.size(1) == 0:
        dataA = dataA
        dataB = dataB
    else:
        dataA = torch.cat((dataA, data1), dim=1)
        dataB = torch.cat((dataB, data2), dim=1)
    def const_proj(evec, M):
        M = torch.diag(M)
        return torch.linalg.pinv(evec.t() @ M @ evec) @ evec.t() @ M
    proj1 = const_proj(evec1, mass1)
    proj2 = const_proj(evec2, mass2)
    # #my new NN search objective

    # _, pred_labels2to1 = diffusion_net.geometry.find_knn(dataB, dataA, k=1, method='cpu_kd')
    # pred_labels2to1 = pred_labels2to1.squeeze(-1)
    
    pred_labels2to1 = nn_query(dataA, dataB) 

    return pred_labels2to1

def compute_correspondence_expanded(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, sqrtM1k, invsqrtM2k, lambda_param=1e-3):
    """
    Computes the functional map correspondence matrix C given features from two shapes.

    Has no trainable parameters.
    """

    sqrtM1k = sqrtM1k.squeeze()
    invsqrtM2k = invsqrtM2k.squeeze()

    feat_x = feat_x.squeeze()
    feat_y = feat_y.squeeze()
    evals_x = evals_x.squeeze()
    evals_y = evals_y.squeeze()
    evecs_trans_x = evecs_trans_x.squeeze()
    evecs_trans_y = evecs_trans_y.squeeze()

    A = evecs_trans_x @ feat_x
    B = evecs_trans_y @ feat_y
    # A and B should be same shape
    k = A.size(0)
    m = A.size(1)

    vec_B = B.T.reshape(m * k, 1).contiguous()

    A_t = A.T.contiguous()
    Ik = torch.eye(k, device=A.device, dtype=torch.float32)

    At_Ik = torch.kron(A_t, Ik)

    lx = torch.diag(evals_x.squeeze(0))
    ly = torch.diag(evals_y.squeeze(0))
    lx_Ik = torch.kron(lx, sqrtM1k)
    Ik_ly = torch.kron(invsqrtM2k, ly)
    Delta = (lx_Ik - Ik_ly)

    first = At_Ik.T @ At_Ik
    second = Delta.T @ Delta
    rhs = At_Ik.T @ vec_B
    op = first + lambda_param * second

    C = torch.linalg.solve(op, rhs)

    return C.reshape(k, k).T


def killnan(tensor):
    if torch.isnan(tensor).any():
        # warning message
        # print(f'Warning: {torch.isnan(tensor).sum()} nan values found')
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor) 
    return tensor

def countnan(tensor):
    return torch.isnan(tensor).sum()


def treatnan(tensor, verts, name):
    # tensor of shape [V, K]
    # verts of shape [V, 3]
    # treat nan as nearest non-nan neighbor
    # we know that nans always come in entire rows
    # so we can do operations for entire rows
    # first find the nan rows
    nan_rows = torch.isnan(tensor).any(dim=1)
    # print warning message if there are nan rows
    if nan_rows.any():
        # print(f'Warning: {nan_rows.sum()} rows have nan values in {name}, replacing with nearest non-nan neighbor')
        # print the row number as well
        print(f'Warning: {nan_rows.sum()} rows have nan values in {name}, replacing with nearest non-nan neighbor')
        print(f'Row numbers: {nan_rows.nonzero(as_tuple=True)[0]}')
    # find the non-nan rows
    non_nan_rows = ~nan_rows
    # find the nearest non-nan neighbor for each nan row
    # this is a tensor of shape [nan_rows, 3]
    nan_verts = verts[nan_rows]
    non_nan_verts = verts[non_nan_rows]
    dists = torch.cdist(nan_verts, non_nan_verts)
    nearest_non_nan_dists, nearest_non_nan_indices = torch.min(dists, dim=1)
    nearest_non_nan_verts = non_nan_verts[nearest_non_nan_indices]
    # replace the nan rows with the nearest non-nan neighbor
    tensor[nan_rows] = tensor[non_nan_rows][nearest_non_nan_indices]
    return tensor



def log_heatmaps(C_gt, Cxy, LB_mask, elas_Cxy_gt, elas_Cxy_pred, elastic_mask, self, data):
    # hmpath = os.path.join(self.opt['path']['results_root'],'heatmaps')
    data_x, data_y = data['first'], data['second']
    name_x, name_y = data_x['name'][0], data_y['name'][0]
    # Create a figure with six subplots in a 2x3 layout
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Plotting the first heatmap
    heatmap1 = axs[0, 0].imshow(C_gt.squeeze(0).detach().cpu().numpy(), cmap='RdYlBu', interpolation='nearest')
    axs[0, 0].set_title('LB C_gt')
    fig.colorbar(heatmap1, ax=axs[0, 0], fraction=0.046, pad=0.04)  # Adjust colorbar size and position

    # Plotting the second heatmap
    heatmap2 = axs[0, 1].imshow(Cxy.squeeze(0).detach().cpu().numpy(), cmap='RdYlBu', interpolation='nearest')
    axs[0, 1].set_title('LB C_pred')
    fig.colorbar(heatmap2, ax=axs[0, 1], fraction=0.046, pad=0.04)  # Adjust colorbar size and position

    # Plotting the third heatmap
    heatmap3 = axs[0, 2].imshow(LB_mask.squeeze(0).detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    axs[0, 2].set_title('LB_mask')
    fig.colorbar(heatmap3, ax=axs[0, 2], fraction=0.046, pad=0.04)  # Adjust colorbar size and position

    # Plotting the fourth heatmap
    heatmap4 = axs[1, 0].imshow(elas_Cxy_gt.squeeze(0).detach().cpu().numpy(), cmap='PRGn', interpolation='nearest')
    axs[1, 0].set_title('elas_C_gt')
    fig.colorbar(heatmap4, ax=axs[1, 0], fraction=0.046, pad=0.04)  # Adjust colorbar size and position

    # Plotting the fifth heatmap
    heatmap5 = axs[1, 1].imshow(elas_Cxy_pred.squeeze(0).detach().cpu().numpy(), cmap='PRGn', interpolation='nearest')
    axs[1, 1].set_title('elas_C_pred')
    fig.colorbar(heatmap5, ax=axs[1, 1], fraction=0.046, pad=0.04)  # Adjust colorbar size and position
    
    # Plotting the sixth heatmap
    heatmap6 = axs[1, 2].imshow(elastic_mask.squeeze(0).detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    axs[1, 2].set_title('elastic_mask')
    fig.colorbar(heatmap6, ax=axs[1, 2], fraction=0.046, pad=0.04)  # Adjust colorbar size and position
    
    plt.savefig(os.path.join(self.opt['path']['results_root'],'heatmaps',  f'{name_x}-{name_y}.jpg'))
    plt.close('all')

     # copy the html to the path
    html_destination = os.path.join(self.opt['path']['results_root'], 'heatmaps', f'all.html')
    if not os.path.exists(html_destination):
        html_path_source = os.path.join('.', 'figures', 'html', self.opt['name']+'.html')
        with open(html_path_source, 'rb') as src:
            with open(html_destination, 'wb') as dest:
                dest.write(src.read())

def get_mask(evals_x, evals_y):
    D = torch.repeat_interleave(evals_x.unsqueeze(1), repeats=evals_x.size(1), dim=1)
    D = (D - torch.repeat_interleave(evals_y.unsqueeze(2), repeats=evals_x.size(1), dim=2)) ** 2

    return D.squeeze()




def concat_basis(data, n, elas_n):
    """
    concat the spectral operators (from LB to Elastic) to specified numbers, intended for mixing up basis Fmap computation

    """
    # everthing has a batch dim
    evals = data['evals']
    # data['evals'] = data['evals'][:, :n]
    # data['evecs'] = data['evecs'][:, :, :n]
    # data['evecs_trans'] = data['evecs_trans'][:, :n, :]

    data['elas_evals'] = torch.concat((data['elas_evals'][:, :elas_n], evals[:, :n]), dim=1)
    data['elas_evecs'] = torch.concat((data['elas_evecs'][:, :, :elas_n], data['evecs'][:, :, :n]), dim=2)

    # recompute the elastic evecs_trans
    mass = data['elas_mass'].squeeze()
    sqrtmass = torch.sqrt(mass)
    evecs = data['elas_evecs'].squeeze()
    def const_proj(evec, sqrtmass):
        # orthogonal projector for elastic basis
        sqrtM = torch.diag(sqrtmass)
        return torch.linalg.pinv(sqrtM @ evec) @ sqrtM
    evecs_trans = const_proj(evecs, sqrtmass)

    data['elas_evecs_trans'] = evecs_trans.unsqueeze(0)
    # recompute the elastic Mk
    Mk = evecs.T @ torch.diag(mass) @ evecs
    data['elas_Mk'] = Mk.unsqueeze(0)

    # recompute sqrtMk and invsqrtMk
    sqrtMk = scipy.linalg.sqrtm(to_numpy(Mk)).real #numerical weirdness from LB
    sqrtMk = torch.tensor(sqrtMk).float().to(Mk.device)
    invsqrtMk = torch.linalg.pinv(sqrtMk)
    data['elas_sqrtMk'] = sqrtMk.unsqueeze(0)
    data['elas_invsqrtMk'] = invsqrtMk.unsqueeze(0)
    return data



def dpfm_fmap_solver(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_=100, resolvant_gamma=0.5, A=None, B=None, alpha=100, C_gt=None):
    def get_mask(evals1, evals2, gamma=0.5, device="cpu"):
        scaling_factor = max(torch.max(evals1), torch.max(evals2))
        evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
        evals_gamma1, evals_gamma2 = (evals1 ** gamma)[None, :], (evals2 ** gamma)[:, None]

        M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
        M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
        return M_re.square() + M_im.square()
    # compute linear operator matrix representation C1 and C2
    evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
    evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

    F_hat = torch.bmm(evecs_trans_x, feat_x)
    G_hat = torch.bmm(evecs_trans_y, feat_y)
    if A is None and B is None:
        A, B = F_hat, G_hat

    D = get_mask(evals_x.flatten(), evals_y.flatten(), resolvant_gamma, feat_x.device).unsqueeze(0)

    A_t = A.transpose(1, 2)
    A_A_t = torch.bmm(A, A_t)
    B_A_t = torch.bmm(B, A_t)

    C_i = []
    for i in range(evals_x.size(1)):
        D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
        C = torch.bmm(torch.inverse(A_A_t + lambda_ * D_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
        C_i.append(C.transpose(1, 2))
    C = torch.cat(C_i, dim=1)

    # optional treatment-----can clean up later
    if C_gt is not None:
        C_i = []
        I = torch.eye(C_gt.size(1)).to(C_gt.device)
        I = I.unsqueeze(0)
        
        for i in range(evals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + lambda_ * D_i + alpha * I), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2) + alpha * C_gt[:, i, :].unsqueeze(1).transpose(1, 2))
            C_i.append(C.transpose(1, 2))
        C = torch.cat(C_i, dim=1)

    return C

# temporary experiment for cleaning up fmap entry for cp2p dpfm for training
def get_clean_C_gt(data, n_fmap, corr_x, corr_y, old_C_gt=None):
    #----------------code from within dpfm
    verts1, faces1, mass1, L1, evals1, evecs1, gradX1, gradY1 = (data["first"]["verts"], data["first"]["faces"], data["first"]['operators']["mass"],
                                                                    data["first"]['operators']["L"], data["first"]['operators']["evals"], data["first"]['operators']["evecs"],
                                                                    data["first"]['operators']["gradX"], data["first"]['operators']["gradY"])
    verts2, faces2, mass2, L2, evals2, evecs2, gradX2, gradY2 = (data["second"]["verts"], data["second"]["faces"], data["second"]['operators']["mass"],
                                                                    data["second"]['operators']["L"], data["second"]['operators']["evals"], data["second"]['operators']["evecs"],
                                                                    data["second"]['operators']["gradX"], data["second"]['operators']["gradY"])
    # let's get rid of all the batch dimensions, assume batch size is always 1 so just index in 0
    verts1, faces1, mass1, L1, evals1, evecs1, gradX1, gradY1 = verts1[0], faces1[0], mass1[0], L1[0], evals1[0], evecs1[0], gradX1[0], gradY1[0]
    verts2, faces2, mass2, L2, evals2, evecs2, gradX2, gradY2 = verts2[0], faces2[0], mass2[0], L2[0], evals2[0], evecs2[0], gradX2[0], gradY2[0]
    # squeeze evecs, mass
    evecs1, evecs2 = evecs1.squeeze(0), evecs2.squeeze(0)
    mass1, mass2 = mass1.squeeze(0), mass2.squeeze(0)
    evals1, evals2 = evals1.squeeze(0), evals2.squeeze(0)
    # predict fmap
    evecs_trans1, evecs_trans2 = evecs1.t()[:n_fmap] @ torch.diag(mass1), evecs2.t()[:n_fmap] @ torch.diag(mass2)
    evals1, evals2 = evals1[:n_fmap], evals2[:n_fmap]
    


    # construct custom use_feat1 and use_feat2 with corr
    n_corr = corr_x.shape[0]
    use_feat1 = torch.zeros((evecs1.shape[0], n_corr)).to(evecs1.device)
    use_feat2 = torch.zeros((evecs2.shape[0], n_corr)).to(evecs1.device)
    use_feat1[corr_x, torch.arange(n_corr)] = 1
    use_feat2[corr_y, torch.arange(n_corr)] = 1
    use_feat1 = use_feat1.unsqueeze(0)
    use_feat2 = use_feat2.unsqueeze(0)
    # weird A and B manipulation---igonred for now
    A = evecs1[corr_x, :50].t()
    B = (evecs1[corr_x, : 50] @ old_C_gt.squeeze(0)).t()
    A = A.unsqueeze(0)
    B = B.unsqueeze(0)
    A=None
    B=None


    # new idea use "perfect features" to get the C_gt

    # first get the names of the two shapes
    name1, name2 = data["first"]["name"][0], data["second"]["name"][0]
    def name2corr(name):
        """input example cat-4"""
        cat, num = name.split('-')
        path = f"../data/SHREC16/cuts/corres/cuts_{cat}_shape_{num}.vts"
        corr = np.loadtxt(path, dtype=int) - 1
        return corr
    corr_1 = name2corr(name1)
    corr_2 = name2corr(name2)

    cat = name1.split('-')[0]
    # get the perfect features
    feat = torch.load(f"cat.pth").squeeze(0)
    feat1 = feat[corr_1] # N, 128
    feat2 = feat[corr_2]

    gt_partiality_mask12 = data['first']['partiality_mask'].squeeze() # N
    gt_partiality_mask21 = data['second']['partiality_mask'].squeeze()

    # only keep overlap values and rest to zero
    feat1 = feat1 * gt_partiality_mask12.unsqueeze(1)
    feat2 = feat2 * gt_partiality_mask21.unsqueeze(1)
    # print(feat1.shape)
    # print(feat1)
    # exit()


    use_feat1 = feat1.unsqueeze(0)
    use_feat2 = feat2.unsqueeze(0)


        
    C_gt = dpfm_fmap_solver(use_feat1, use_feat2, evals1, evals2, evecs_trans1, evecs_trans2, lambda_=5e-3, A=A, B=B, alpha=1e-5, C_gt=old_C_gt)

    # with the new perfect features
    C_gt = dpfm_fmap_solver(use_feat1, use_feat2, evals1, evals2, evecs_trans1, evecs_trans2, lambda_=100, A=A, B=B, alpha=0, C_gt=old_C_gt)


    return C_gt