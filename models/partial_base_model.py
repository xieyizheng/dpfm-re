from .base_model import BaseModel
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from utils import get_root_logger
from utils.logger import AvgTimer
from utils.tensor_util import to_numpy
import pickle

class PartialBaseModel(BaseModel):
    def __init__(self, opt):
        super(PartialBaseModel, self).__init__(opt)
        
    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            tb_logger (tensorboard logger): Tensorboard logger.
            update (bool): update best metric and best model. Default True
        """
        self.eval()

        # save results
        metrics_result = {}

        # geodesic errors
        geo_errors = []

        # p2p list
        p2p_list = []

        # val name pairs
        val_name_pairs = []

        # overlap_score12 list
        overlap_score12_list = []

        # overlap_score21 list
        overlap_score21_list = []

        # iou lists
        iou1_list = []
        iou2_list = []

        # one iteration
        timer = AvgTimer()
        pbar = tqdm(dataloader)
        for index, data in enumerate(pbar):
            p2p, Pyx, Cxy, overlap_score12, overlap_score21 = self.validate_single(data, timer)
            p2p, Pyx, Cxy, overlap_score12, overlap_score21 = to_numpy(p2p), to_numpy(Pyx), to_numpy(Cxy), to_numpy(overlap_score12), to_numpy(overlap_score21)
            if 'geo_error' in self.metrics:
                data_x, data_y = data['first'], data['second']
                # get geodesic distance matrix
                if 'dist' in data_x:
                    dist_x = data_x['dist']
                else:
                    dist_x = torch.cdist(data_x['verts'], data_x['verts'])

                # get gt correspondence
                corr_x = data_x['corr']
                corr_y = data_y['corr']

                # convert torch.Tensor to numpy.ndarray
                dist_x = to_numpy(dist_x)
                corr_x = to_numpy(corr_x)
                corr_y = to_numpy(corr_y)

                # compute geodesic error
                geo_err = self.metrics['geo_error'](dist_x, corr_x, corr_y, p2p, return_mean=False)

                avg_geo_err = geo_err.mean()
                geo_errors += [geo_err]

                # save p2p
                p2p_list.append(p2p)

                # save name pairs
                val_name_pairs.append((data_x['name'][0], data_y['name'][0]))

                # save overlap scores
                overlap_score12_list.append(overlap_score12)
                overlap_score21_list.append(overlap_score21)

                # compute IoU scores
                gt_overlap12 = to_numpy(data_x['partiality_mask'])
                gt_overlap21 = to_numpy(data_y['partiality_mask'])
                pred_overlap12 = overlap_score12 > 0.5
                pred_overlap21 = overlap_score21 > 0.5
                
                intersection1 = np.logical_and(gt_overlap12, pred_overlap12)
                union1 = np.logical_or(gt_overlap12, pred_overlap12)
                iou1 = np.sum(intersection1) / np.sum(union1)
                iou1_list.append(iou1)

                intersection2 = np.logical_and(gt_overlap21, pred_overlap21)
                union2 = np.logical_or(gt_overlap21, pred_overlap21)
                iou2 = np.sum(intersection2) / np.sum(union2)
                iou2_list.append(iou2)

                name_x, name_y = data_x['name'][0], data_y['name'][0]
                logger = get_root_logger()
                miou1_so_far = np.mean(iou1_list)
                miou2_so_far = np.mean(iou2_list)
                logger.info(f'avg err:{np.concatenate(geo_errors).mean():.4f} | err:{avg_geo_err:.4f} | iou1:{iou1:.4f} | iou2:{iou2:.4f} | miou1:{miou1_so_far:.4f} | miou2:{miou2_so_far:.4f} | {name_x} {name_y}')
                
                # plot pck per pair
                if 'plot_pck_per_pair' in self.metrics:
                    # compute compare pck plot
                    fig =self.metrics['plot_pck_per_pair']([geo_err], [f"Err: {avg_geo_err:.5f}"], threshold=0.20, steps=40)
                    # save plt
                    name_x, name_y = data_x['name'][0], data_y['name'][0]
                    # print(name_x, name_y, avg_geo_err)
                    plt.savefig(os.path.join(self.opt['path']['results_root'],'pcks',  f'{name_x}-{name_y}.jpg'))
                    plt.close('all')

        logger = get_root_logger()
        logger.info(f'Avg time: {timer.get_avg_time():.4f}')

        # compute mean IoU scores
        miou1 = np.mean(iou1_list)
        miou2 = np.mean(iou2_list)
        logger.info(f'miou1: {miou1:.4f} | miou2: {miou2:.4f}')

        # save geo_errors before concatenating
        if self.opt['val'].get('save_geo_errors', False):
            # use pickle to save geo_errors because it is a list of np.ndarray of different sizes
            with open(os.path.join(self.opt['path']['results_root'], 'geo_errors.pkl'), 'wb') as f:
                pickle.dump(geo_errors, f)
            # save name pairs use pickle as well
            with open(os.path.join(self.opt['path']['results_root'], 'val_name_pairs.pkl'), 'wb') as f:
                pickle.dump(val_name_pairs, f)
            # save p2p use pickle as well
            with open(os.path.join(self.opt['path']['results_root'], 'p2p.pkl'), 'wb') as f:
                pickle.dump(p2p_list, f)
            # save overlap scores use pickle as well
            with open(os.path.join(self.opt['path']['results_root'], 'overlap_score12.pkl'), 'wb') as f:
                pickle.dump(overlap_score12_list, f)
            with open(os.path.join(self.opt['path']['results_root'], 'overlap_score21.pkl'), 'wb') as f:
                pickle.dump(overlap_score21_list, f)
            # save iou1 use pickle as well
            with open(os.path.join(self.opt['path']['results_root'], 'iou1.pkl'), 'wb') as f:
                pickle.dump(iou1_list, f)
            # save iou2 use pickle as well
            with open(os.path.join(self.opt['path']['results_root'], 'iou2.pkl'), 'wb') as f:
                pickle.dump(iou2_list, f)
            
        # entire validation results
        if len(geo_errors) != 0:
            geo_errors = np.concatenate(geo_errors)
            avg_geo_error = geo_errors.mean()
            metrics_result['avg_error'] = avg_geo_error
            metrics_result['miou1'] = miou1
            metrics_result['miou2'] = miou2

            auc, fig, pcks = self.metrics['plot_pck'](geo_errors, threshold=self.opt['val'].get('auc', 0.25))
            metrics_result['auc'] = auc

            # Plot IoU curve
            iou_fig = self.metrics['plot_iou_curve'](iou2_list)

            if tb_logger is not None:
                step = self.curr_iter // self.opt['val']['val_freq']
                tb_logger.add_figure('pck', fig, global_step=step)
                tb_logger.add_figure('iou_curve', iou_fig, global_step=step)
                tb_logger.add_scalar('val auc', auc, global_step=step)
                tb_logger.add_scalar('val avg error', avg_geo_error, global_step=step)
                tb_logger.add_scalar('val miou1', miou1, global_step=step)
                tb_logger.add_scalar('val miou2', miou2, global_step=step)
            else:
                fig.savefig(os.path.join(self.opt['path']['results_root'], 'pck.png'))
                iou_fig.savefig(os.path.join(self.opt['path']['results_root'], 'iou_curve.png'))
                np.save(os.path.join(self.opt['path']['results_root'], 'pck.npy'), pcks)
            plt.close('all')

            # display results
            logger = get_root_logger()
            logger.info(f'Val auc: {auc:.4f}')
            logger.info(f'Val avg error: {avg_geo_error:.4f}')

            # update best model state dict
            if update and (self.best_metric is None or (metrics_result['avg_error'] < self.best_metric)):
                self.best_metric = metrics_result['avg_error']
                self.best_networks_state_dict = self._get_networks_state_dict()
                logger.info(f'Best model is updated, average geodesic error: {self.best_metric:.4f}')
                # save current best model
                self.save_model(net_only=False, best=True, save_filename="best.pth")

        # train mode
        self.train()