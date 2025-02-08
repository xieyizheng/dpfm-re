# DPFM-RE[produce]  
📌 **Reproduced DPFM with close results to the original paper.**  

<p align="center">
  <img src="assets/teaser_texture.jpg" width="45%" style="display: inline-block;" />
  <img src="assets/teaser_pck.jpg" width="45%" style="display: inline-block;" />
</p>

## Scope  
- **Datasets**: `cuts`, `holes`, `cp2p`, `pfarm`  
- **Evaluation Metrics**:  
  - Geodesic Error  
  - PCK Curve  
  - mIoU & mIoU Curve  
- **Visualizations**: A script for generating figures  

## ✅ Done  
- fix nn_interpolate bug
- fix nce_loss bug
- `cp2p` test  
- Geodesic Error  
- PCK Curve  
- mIoU & mIoU Curve  
- visualization script  
- fix overlap loss bug
- Training script
- cuts holes configs
- custom collate function
- refactor diffsion file loading (optional: use evecs from dataset, load diff from verts and faces)
- augmentation code
- xyz augmentation transformation refactors:
- verts as unique identifier for loading shape properties
- xyz as the thing to freely transfrom, augment, anywhere in the pipeline, dataset, model, diffusion net, etc. 
- preprocess script with simple iteration
- dataset upload (shrec16 from ulrssm, cp2p and pfarm are mit licensed)
- a cache wrapper func for easy cache usage
- clean up the shape quantity caching and loading logic
- fix random seed for xyz test augmentations for consistent results
apparently test augmentations reproducibility and variability is not easy to implement....
solution: use deterministic hash to get reproducible seed for each pair, just like our cache logic.
- refactor diffusionnet
- discovered a bug in ulrssm diffusionnet, the gradient rotation is not applied correctly. We stick to original official diffusionnet implementation from nmwsharp and dpfm for the future. 
- refactor diffusionnet, dpfm signatures
- will not do it. optional preprocess script with dataloader to enable parallel preprocess with multiple workers: tried it, but there is no real difference on my machine, so rather keep it simple
- save best model logic

## 🔨 TODO  
- propogate shape dataset loading logic to other datasets
- pfarm
- evecs number balance in visualization
- val loss logging
- cuts holes ckpts
- better readme (acknowledgements)

