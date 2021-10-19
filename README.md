# Point-Based Modeling of Human Clothing

This is an official PyTorch code repository of the paper "Point-Based Modeling of Human Clothing" (accepted to ICCV, 2021).

<p align="center"><img src="static/vton_all.gif"></p>

<p align="center">
  <b>
  <a href="https://openaccess.thecvf.com/content/ICCV2021/html/Zakharkin_Point-Based_Modeling_of_Human_Clothing_ICCV_2021_paper.html">Paper</a> 
  | <a href="https://saic-violet.github.io/point-based-clothing">Project page</a>
  | <a href="https://youtu.be/kFrAu415kDU">Video</a>
    </b>
</p>

## Setup

### Build docker

- Prerequisites: your nvidia driver should support cuda 10.2, Windows or Mac are not supported.
- Clone repo:
  - `git clone https://github.com/izakharkin/point_based_clothing.git`
  - `cd point_based_clothing`
  - `git submodule init && git submodule update`
- Docker setup:
  - [Install docker engine](https://docs.docker.com/engine/install/ubuntu/)
  - [Install nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - [Set](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#daemon-configuration-file) nvidia your default runtime for docker
  - Make docker run without sudo: create docker group and add current user to it: 
    ```
    sudo groupadd docker
    sudo usermod -aG docker $USER
    ```
  - **Reboot**
- Download [`10_nvidia.json`](https://gitlab.com/nvidia/container-images/opengl/-/blob/2dba242a538fdaa558c5f87017a7cf63eb016582/glvnd/runtime/10_nvidia.json) and place it in the `docker/` folder
- Create docker image: 
  - Build on your own: [run](./docker) 2 commands
- Inside the docker container: `source activate pbc`

## Download data

- Download the SMPL neutral model from [SMPLify project page](https://smplify.is.tue.mpg.de/login.php): 
  - Register, go to the `Downloads` section, download `SMPLIFY_CODE_V2.ZIP`, and unpack it;
  - Move `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `data/smpl_models/SMPL_NEUTRAL.pkl`.
- Download models checkpoints (~570 Mb): [Google Drive](https://drive.google.com/file/d/16QFuHhou_C4EY6GvKCgbJvgbkIycITHw/view?usp=sharing) and place them to the `checkpoints/` folder;
- Download a sample data we provide to check the appearance fitting (~480 Mb): [Google Drive](https://drive.google.com/file/d/13ma8J0-ah4sVn0uH_hYSoJ_GLMxFpoQa/view?usp=sharing), unpack it, and place `psp/` folder to the `samples/` folder.

## Run

We provide scripts for *geometry* fitting and inference and *appearance* fitting and inference.

### Geometry (outfit code)

#### Fitting

To fit a style outfit code to a single image one can run:
```
python fit_outfit_code.py --config_name=outfit_code/psp
```

The learned outfit codes are saved to `out/outfit_code/outfit_codes_<dset_name>.pkl` by default. The visualization of the process is in `out/outfit_code/vis_<dset_name>/`:

* Coarse fitting stage: four outfit codes initialized randomly and being optimized simultaneosly.

<p align="center">
  <img src="static/outfit_code_fitting_coarse.gif" alt="outfit_code_fitting_coarse">
</p>

* Fine fitting stage: mean of found outfit codes is being optimized further to possibly imrove the reconstruction.

<p align="center">
  <img src="static/outfit_code_fitting_fine.gif" alt="outfit_code_fitting_fine" width="224px">
</p>

**Note:** `visibility_thr` hyperparameter in `fit_outfit_code.py` may affect the quality of result point cloud (e.f. make it more sparse). F
