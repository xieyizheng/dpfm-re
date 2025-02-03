import torch
import torch.nn.functional as F

from .partial_base_model import PartialBaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from utils.dpfm_util import cache_sample_idx, cache_operators
import os

@MODEL_REGISTRY.register()
class DPFM_Model(PartialBaseModel):
    def __init__(self, opt):
        super(DPFM_Model, self).__init__(opt)

    def feed_data(self, data):
        # ----------------- stich code to be cleaned up later------------------------
        cache_dir = self.data_root
        cache_dir = os.path.join(cache_dir, 'diffusion')
        cache_operators(data, cache_dir=cache_dir)
        n_fmap = self.opt["networks"]['dpfm_net']['cfg']['fmap']['n_fmap']
        cross_sampling_ratio = self.opt["networks"]['dpfm_net']['cfg']['attention']['cross_sampling_ratio']
        cache_sample_idx(data, cross_sampling_ratio=cross_sampling_ratio, cache_dir=cache_dir) # for caching opereation can use 1.0 and afterwards will be flexible for any ratio
        data = to_device(data, self.device)

        
        #----------------------------DPFM forward pass---------------------------------------
        C_xy, overlap_score12, overlap_score21, use_feat1, use_feat2 = self.networks['dpfm_net'](data)

        # ----------------------------loss calculation----------------------------------------
        # get spectral operators
        evecs_x = data['first']['operators']['evecs'].squeeze()[:,:n_fmap]
        evecs_y = data['second']['operators']['evecs'].squeeze()[:,:n_fmap]
        mass_y = data['second']['operators']['mass'].squeeze()
        evecs_trans_y = evecs_y.t() @ torch.diag(mass_y)
        # get gt correspondence
        corr_x = data['first']['corr'][0]
        corr_y = data['second']['corr'][0]

        # partiality map groud truth
        if 'partiality_mask' in data['first'].keys(): # for cp2p when partiality mask is provided in the .map file
            gt_partiality_mask12 = data['first']['partiality_mask'].squeeze()
            gt_partiality_mask21 = data['second']['partiality_mask'].squeeze()
        else:
            gt_partiality_mask12 = torch.zeros((evecs_x.shape[0])).long().to(self.device)
            gt_partiality_mask12[corr_x[corr_x != -1]] = 1
            gt_partiality_mask21 = torch.zeros((evecs_y.shape[0])).long().to(self.device)
            gt_partiality_mask21[corr_y[corr_y != -1]] = 1

        # calculate ground truth functional map
        P = torch.zeros((evecs_y.shape[0], evecs_x.shape[0])).to(self.device) # [Ny, Nx]
        P[corr_y, corr_x] = 1
        C_gt = evecs_trans_y.squeeze() @ P @ evecs_x
        C_gt = C_gt.unsqueeze(0)

        # loss
        fmap_loss, acc_loss, nce_loss = self.losses["dpfm_loss"](C_gt, C_xy, corr_x, corr_y, use_feat1, use_feat2,
                             overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

        self.loss_metrics = {'fmap_loss': fmap_loss, 'acc_loss': acc_loss, 'nce_loss': nce_loss}




    def validate_single(self, data, timer):
        # start record
        timer.start()

        # ----------------- stich code to be cleaned up later------------------------
        cache_dir = self.data_root
        cache_dir = os.path.join(cache_dir, 'diffusion')
        cache_operators(data, cache_dir=cache_dir)
        n_fmap = self.opt["networks"]['dpfm_net']['cfg']['fmap']['n_fmap']
        cross_sampling_ratio = self.opt["networks"]['dpfm_net']['cfg']['attention']['cross_sampling_ratio']
        cache_sample_idx(data, cross_sampling_ratio=cross_sampling_ratio, cache_dir=cache_dir) # for caching opereation can use 1.0 and afterwards will be flexible for any ratio

        data = to_device(data, self.device)

        #----------------------------DPFM forward pass---------------------------------------
        Cxy, overlap_score12, overlap_score21, use_feat1, use_feat2 = self.networks['dpfm_net'](data)


        # get spectral operators
        evecs_x = data['first']['operators']['evecs'].squeeze()[:,:n_fmap] # the reason we use operators is because it's two different eigendecomposed vectors with the fmap.....
        evecs_y = data['second']['operators']['evecs'].squeeze()[:,:n_fmap]
        mass_y = data['second']['operators']['mass'].squeeze()
        evecs_trans_y = evecs_y.t() @ torch.diag(mass_y)

        mass_x = data['first']['operators']['mass'].squeeze()
        evecs_trans_x = evecs_x.t() @ torch.diag(mass_x)


        #-------------------------recover point-to-point map-----------------------------------
        Cxy = Cxy.squeeze()

        # convert functional map to point-to-point map
        p2p = fmap2pointmap(Cxy, evecs_x, evecs_y)

        # compute Pyx from functional map
        Pyx = evecs_y @ Cxy @ evecs_trans_x

        # finish record
        timer.record()


        return p2p, Pyx, Cxy, overlap_score12, overlap_score21




    