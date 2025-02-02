import numpy as np
import matplotlib.pyplot as plt
from utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    """
    Calculate the geodesic error between predicted correspondence and gt correspondence

    Args:
        dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
        corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
        corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
        p2p (np.ndarray): Point-to-point map (shape y -> shape x). shape [Vy]
        return_mean (bool, optional): Average the geodesic error. Default True.
    Returns:
        avg_geodesic_error (np.ndarray): Average geodesic error.
    """
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err


@METRIC_REGISTRY.register()
def plot_pck(geo_err, threshold=0.20, steps=40):
    """
    plot pck curve and compute auc.
    Args:
        geo_err (np.ndarray): geodesic error list.
        threshold (float, optional): threshold upper bound. Default 0.15.
        steps (int, optional): number of steps between [0, threshold]. Default 30.
    Returns:
        auc (float): area under curve.
        fig (matplotlib.pyplot.figure): pck curve.
        pcks (np.ndarray): pcks.
    """
    assert threshold > 0 and steps > 0
    geo_err = np.ravel(geo_err)
    thresholds = np.linspace(0., threshold, steps)
    pcks = []
    for i in range(thresholds.shape[0]):
        thres = thresholds[i]
        pck = np.mean((geo_err <= thres).astype(float))
        pcks.append(pck)
    pcks = np.array(pcks)
    # compute auc
    auc = np.trapz(pcks, np.linspace(0., 1., steps))

    # display figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(thresholds, pcks, 'r-')
    ax.set_xlim(0., threshold)
    fig = plot_pck_multiple([geo_err], ['err'], threshold=threshold)
    return auc, fig, pcks


def get_pck(geo_err, threshold=0.20, steps=80):
    """
    plot pck curve and compute auc.
    Args:
        geo_err (np.ndarray): geodesic error list.
        threshold (float, optional): threshold upper bound. Default 0.15.
        steps (int, optional): number of steps between [0, threshold]. Default 30.
    Returns:
        auc (float): area under curve.
        fig (matplotlib.pyplot.figure): pck curve.
        pcks (np.ndarray): pcks.
    """
    assert threshold > 0 and steps > 0
    geo_err = np.ravel(geo_err)
    thresholds = np.linspace(0., threshold, steps)
    pcks = []
    for i in range(thresholds.shape[0]):
        thres = thresholds[i]
        pck = np.mean((geo_err <= thres).astype(float))
        pcks.append(pck)
    pcks = np.array(pcks)
    # compute auc
    auc = np.trapz(pcks, np.linspace(0., 1., steps))

    
    return pcks

def plot_pcks(pcks, labels, title="", threshold=0.20, steps=80):
    thresholds = np.linspace(0., threshold, steps)
    # display figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    
    ax.set_ylabel('% correspondences', labelpad=10)
    ax.set_xlabel('geodesic error', labelpad=10)
    
    ax.set_xlim((0, 0.09))
    ax.set_ylim((0, 102))
    for pck, label in zip(pcks, labels):
        ax.plot(thresholds, pck*100, label=label, linewidth=3)
    ax.set_xlim(0., threshold)
    ax.set_title(title)
    plt.grid()
    fig.legend()
    return fig
def single_pck(geo_err, label):
    pck = get_pck(geo_err)
    plot_pcks([pck],[label])

@METRIC_REGISTRY.register()
def plot_pck_multiple(geo_errs, labels, title='', threshold=0.20, steps=80):
    return plot_pcks([get_pck(geo_err, threshold, steps) for geo_err in geo_errs], labels, title, threshold, steps)

@METRIC_REGISTRY.register()
def plot_iou_curve(iou_list, threshold_steps=10):
    """Plot IoU curve at different thresholds.
    
    Args:
        iou_list (list): List of IoU values
        threshold_steps (int, optional): Number of threshold steps. Default 10.
    Returns:
        fig (matplotlib.pyplot.figure): IoU curve plot
    """
    thresholds = np.linspace(0, 1.0, threshold_steps)
    iou_values = []
    
    for threshold in thresholds:
        # Calculate IoU at each threshold
        iou_at_threshold = np.mean(np.array(iou_list) >= threshold)
        iou_values.append(iou_at_threshold * 100)  # Convert to percentage
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(thresholds, iou_values, linewidth=3)
    ax.set_xlabel('IoU threshold')
    ax.set_ylabel('% pairs')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 100)
    
    plt.grid(True)
    
    return fig
