import argparse
from tqdm import tqdm
from os import path as osp
from datasets import build_dataset
from utils.options import parse

def preprocess_dataset(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=False)
    
    # Process each dataset in the config
    for dataset_name, dataset_opt in opt["datasets"].items():
        if isinstance(dataset_opt, int):  # skip batch_size, num_worker entries
            continue
        # Print dataset config
        print("\n" + "="*50)
        print(f"Dataset: {dataset_name}")
        print(f"Type: {dataset_opt['type']}")
        print(f"Name: {dataset_opt['name']}")
        print(f"Data root: {dataset_opt['data_root']}")
        print(f"Return evecs: {dataset_opt.get('return_evecs', False)}")
        print(f"Return faces: {dataset_opt.get('return_faces', False)}")
        print(f"Num evecs: {dataset_opt.get('num_evecs', 0)}")
        print(f"Return corr: {dataset_opt.get('return_corr', False)}")
        print(f"Return dist: {dataset_opt.get('return_dist', False)}")
        print("="*50 + "\n")
        
        # Build dataset
        print(f"Building dataset {dataset_name}...")
        test_set = build_dataset(dataset_opt)
        print(f"Dataset size: {len(test_set)}")
        
        # Iterate through dataset
        print(f"Iterating through dataset {dataset_name}...")
        for i in tqdm(range(len(test_set))):
            data = test_set[i]
            # we do nothing other than iterating through the dataset since our dataset compute and cache all data needed on the fly

if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    preprocess_dataset(root_path)
