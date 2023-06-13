# FlowCam: Training Generalizable 3D Radiance Fields without Camera Poses via Pixel-Aligned Scene Flow
### [Project Page](https://cameronosmith.github.io/flowcam) | [Paper](https://arxiv.org/abs/2306.00180)  | [Pretrained Models](https://drive.google.com/drive/folders/1t7vmvBg9OAo4S8I2zjwfqhL656H1r2JP?usp=sharing)

[Cameron Smith](https://cameronosmith.github.io/),
[Yilun DU](https://yilundu.github.io/),
[Ayush Tewari](https://ayushtewari.com),
[Vincent Sitzmann](https://vsitzmann.github.io/)

MIT

This is the official implementation of the paper "FlowCam: Training Generalizable 3D Radiance Fields without Camera Poses via Pixel-Aligned Scene Flow".

<img src="https://cameronosmith.github.io/flowcam/img/pipeline.png"  width="800">

## High-Level structure
The code is organized as follows:
* models.py contains the model definition
* run.py contains a generic argument parser which creates the model and dataloaders for both training and evaluation
* train.py and eval.py contains train and evaluation loops
* mlp_modules.py and conv_modules.py contain common MLP and CNN blocks
* vis_scripts.py contains plotting and wandb logging code 
* renderer.py implements volume rendering helper functions
* geometry.py implements various geometric operations (projections, 3D lifting, rigid transforms, etc.)
* data contains a list of dataset scripts
* demo.py contains a script to run our model on any image directory for pose estimates. See the file header for an example on running it.

## Reproducing experiments

See `python run.py --help` for a list of command line arguments. 
An example training script on CO3D-Hydrants is `python train.py --dataset hydrant --vid_len 8 --batch_size 2 --online --name hydrants_flowcam.`
Similarly, replace `--dataset hydrants` with any of `[realestate,kitti,10cat]` for training on RealEstate10K, KITTI, or CO3D-10Category.

Use the `--online` flag for summaries to be logged to your wandb account. 

## Environment variables

We use environment variables to set the dataset and logging paths, though you can easily hardcode the paths in each respective dataset script. Specifically, we use the environment variables `CO3D_ROOT, RE10K_IMG_ROOT, RE10K_POSE_ROOT, KITTI_ROOT, and LOGDIR`. For instance, you can add the line `export CO3D_ROOT="/nobackup/projects/public/facebook-co3dv2"` to your `.bashrc`.

## Data

The KITTI dataset we use can be downloaded here: https://www.cvlibs.net/datasets/kitti/raw_data.php
Instructions for downloading the RealEstate10K dataset can be found here: https://github.com/yilundu/cross_attention_renderer/blob/master/data_download/README.md
We use the V2 version of the CO3D dataset, which can be downloaded here: https://github.com/facebookresearch/co3d

### Coordinate and camera parameter conventions
This code uses an "OpenCV" style camera coordinate system, where the Y-axis points downwards (the up-vector points in the negative Y-direction), the X-axis points right, and the Z-axis points into the image plane.

### Citation
If you find our work useful in your research, please cite:
```
@misc{smith2023flowcam,
      title={FlowCam: Training Generalizable 3D Radiance Fields without Camera Poses via Pixel-Aligned Scene Flow}, 
      author={Cameron Smith and Yilun Du and Ayush Tewari and Vincent Sitzmann},
      year={2023},
      eprint={2306.00180},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Contact
If you have any questions, please email Cameron Smith at omid.smith.cameron@gmail.com or open an issue.
