import pickle
import argparse
import numpy as np
from tqdm import tqdm
import polyscope as ps
from os import path as osp
from PIL import Image, ImageEnhance
from datasets import build_dataset
from utils.options import parse
from utils.geometry_util import torch2np
from utils.texture_util import *
from utils.visualization_util import *

ps.set_allow_headless_backends(True) 
ps.init()
ps.set_ground_plane_mode("none")
ps.set_view_projection_mode("orthographic")
ps.set_SSAA_factor(4)
ps.set_window_size(3840, 2160)

def visualize_pipeline(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=False)
    ours_results_root = opt["path"]["results_root"]
    ours_p2p_list = pickle.load(open(osp.join(ours_results_root, "p2p.pkl"), "rb"))
    overlap_score12_list = pickle.load(open(osp.join(ours_results_root, "overlap_score12.pkl"), "rb"))
    overlap_score21_list = pickle.load(open(osp.join(ours_results_root, "overlap_score21.pkl"), "rb"))
    # iou1_list = pickle.load(open(osp.join(ours_results_root, "iou1.pkl"), "rb"))
    iou2_list = pickle.load(open(osp.join(ours_results_root, "iou2.pkl"), "rb"))
    # Example of loading baseline results (commented out)
    # baseline_results_root = "path/to/baseline/results"
    # baseline_p2p_list = pickle.load(open(osp.join(baseline_results_root, "p2p.pkl"), "rb"))
    # baseline_overlap12_list = pickle.load(open(osp.join(baseline_results_root, "overlap_score12.pkl"), "rb")) 
    # baseline_overlap21_list = pickle.load(open(osp.join(baseline_results_root, "overlap_score21.pkl"), "rb"))
    vis_manager = VisualizationManager(ours_results_root)
    ps.set_user_callback(vis_manager.callback)
    if not osp.exists(osp.join(ours_results_root, "polyscope_visualization")):
        os.makedirs(osp.join(ours_results_root, "polyscope_visualization"))
    dataset_opt = opt["datasets"].popitem()[1]
    dataset_opt.update({"return_dist": False, "return_elas_evecs": False, "cache": False, "num_evecs": 200})
    test_set = build_dataset(dataset_opt)
    orient_calib_R = orientation_calibration_by_dataset(test_set)

    # color and texture
    texture = np.array(Image.open("assets/7.png").convert("RGB")) / 255.0
    texture[-1, 0, :] = 1  # Set corner 0,0 to white to indicate non_overlapping region

    # iterate from best iou to worst iou
    sorted_indices = np.argsort(iou2_list)[::-1]  # Sort by IoU in descending order
    
    for i in tqdm(sorted_indices):
        data = test_set[i]
        data_x, data_y = data["first"], data["second"]
        verts_x, verts_y = torch2np(data_x["verts"]) @ orient_calib_R, torch2np(data_y["verts"]) @ orient_calib_R
        faces_x, faces_y = torch2np(data_x["faces"]), torch2np(data_y["faces"])
        evecs_x, evecs_y = torch2np(data_x["evecs"]), torch2np(data_y["evecs"])
        evecs_trans_x, evecs_trans_y = torch2np(data_x["evecs_trans"]), torch2np(data_y["evecs_trans"])

        # Define corres and overlaps from gt, ours, and optionally another baseline
        correspondence_methods = {
            "gt": {
                "corr_x": torch2np(data_x["corr"]),
                "corr_y": torch2np(data_y["corr"]),
                "overlap12": torch2np(data_x["partiality_mask"]),
                "overlap21": torch2np(data_y["partiality_mask"])
            },
            # Add more methods here as needed, e.g.:
            # "baseline1": { ... },
            # "baseline2": { ... }
            "ours": {
                "corr_x": ours_p2p_list[i],
                "corr_y": np.arange(len(ours_p2p_list[i])),
                "overlap12": overlap_score12_list[i] > 0.5,
                "overlap21": overlap_score21_list[i] > 0.5
            },
        }

        # Register meshes for all methods
        vis_manager.register_meshes(verts_x, verts_y, faces_x, faces_y, list(correspondence_methods.keys()))
        
        # CAUTION: This is a "naughty" implementation: a better implementation should be thought of in the future
        # But this works for our visualization purposes ; 
        def retexture_callback(R1, R2):
            rotated_verts_x = verts_x @ R1
            rotated_verts_y = verts_y @ R2
            for prefix, method_data in correspondence_methods.items():
                corner_uv_x, corner_uv_y = compute_partial_texture_mapping(
                    rotated_verts_x, rotated_verts_y, faces_x, faces_y, evecs_x, evecs_trans_x,
                    method_data["corr_x"], method_data["corr_y"],
                    method_data["overlap12"], method_data["overlap21"]
                )
                vis_manager.add_texture_visualization(prefix, corner_uv_x, corner_uv_y, texture)
        
        # Set up callback and do initial texturing with current state rotations
        vis_manager.texture_callback = retexture_callback
        R1 = get_rotation_matrix(vis_manager.rot_x1, vis_manager.rot_y1, vis_manager.rot_z1)
        R2 = get_rotation_matrix(vis_manager.rot_x2, vis_manager.rot_y2, vis_manager.rot_z2)
        retexture_callback(R1, R2)  # Initial texture with current state rotations

        ps.show()
        ps.frame_tick()
        # screenshot to save
        ps.screenshot(osp.join(ours_results_root, "polyscope_visualization", f"{i}_{data_x['name']}_{data_y['name']}.png"))

    print(f"Visualization saved to {osp.join(ours_results_root, 'polyscope_visualization')}")

if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    visualize_pipeline(root_path)
