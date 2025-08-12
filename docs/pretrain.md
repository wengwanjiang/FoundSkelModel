# Pretrain

## Data Preparation

### NTU RGB-D 

- Download raw [NTU-RGB+D 60 and 120 skeleton data](https://github.com/shahroudy/NTURGB-D) and save to ./data folder.
```
.
└──data/
   ├── nturgbd_raw/
   │    ├── nturgb+d_skeletons/
   |    ├── ...
   | 	└── samples_with_missing_skeletons.txt
   └── nturgbd_raw_120/
        ├── nturgb+d_skeletons/
        ├── ...
        └── samples_with_missing_skeletons.txt
```
- Preprocess data with `data_gen/ntu_gendata.py`.
```python
cd data_gen
python ntu_gendata.py --data_path ./data/nturgbd_raw # for NTU60
python ntu_gendata.py --data_path ./data/nturgbd_raw_120 # for NTU120
```

### UAV-Human
- Download raw [UAV-Human skeleton data](https://sutdcv.github.io/uav-human-web/) and save to ./data folder.
```
.
└──data/
   └── uav/
       └──Skeleton/
           ├──P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
           ├──P000S00G10B10H10UC022000LC021000A000R0_08241716.txt   
           └── ...      
```
- Preprocess data with `data_gen/uav_gendata.py`.
```python
cd data_gen
python uav_gendata.py --data_path ./data/uav/Skeleton # for UAV v1 and v2
```

### PKU-MMD

- Download raw [PKU-MMD skeleton data](https://github.com/ECHO960/PKU-MMD) and save to ./data folder.
```
.
└──data/
   ├── PKUMMDv1/
   │   ├──Data/
   │   ├──Label/
   │   └──Split/
   └──PKUMMDv2/
       ├──Datav2/
       ├──Labelv2/
       └──Spliv2/

```
- Preprocess data with `data_gen/pku_gendata.py`.
```python
cd data_gen
python pku_gendata.py --data_path ./data/PKUMMDv1 # for PKU-MMD v1
python pku_gendata.py --data_path ./data/PKUMMDv2 # for PKU-MMD v2
```

### NTU RGB-D 2D 

- The NTURGB+D 2D detection results are provided by [pyskl](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md) using [HRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation).
- Download [ntu60_hrnet.pkl](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl) and [ntu120_hrnet.pkl](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl) to `data/NTU2D/`.



You can also download the preprocessed data for PKU-MMD and UAV from [here](https://1drv.ms) and unzip it to `data/`.





The processed directory tree should look like this:

```
.
└── data/
    ├── NTU60-2D/
    │   ├──xview/
    │   └──xsub/
    ├── NTU120-2D/
    │   ├──xsetup/
    │   └──xsub/
    ├── NTU60-3D/
    │   ├──xview/
    │   └──xsub/
    ├── NTU120-3D/
    │   ├──xsetup/
    │   └──xsub/
    ├── UAV-2D/
    │   ├──v1/
    │   └──v2/
    ├── PKUMMD/
    │   ├──v1/
    │   └──v2/
    └── ...
```



## Pretrain

```bash
# ============================================
# Pretraining script for NTU60 2D dataset (Cross-View Protocol)
# ============================================

# GPUs to be used (order matters if using DataParallel or DistributedDataParallel)
gpu=3,2,1,0  # $1: GPU device IDs

# Experiment directory name (used for saving checkpoints and logs)
dir=ntu2d60_xv_j_dste  # $2: experiment identifier

# Create checkpoint directory if it doesn't exist
mkdir ./checkpoint/${dir}

# ------------------------------------------------------------
# Launch pretraining
# ------------------------------------------------------------
CUDA_VISIBLE_DEVICES="$gpu" python pretrain.py \
    --lr 0.0001 \                  # Learning rate
    --batch-size 384 \             # Batch size
    --schedule 351 \               # Learning rate schedule step (epoch number)
    --epochs 451 \                 # Total number of training epochs
    --moda joint \                 # Modality: "joint" indicates using joint coordinates
    --checkpoint-path ./checkpoint/${dir} \  # Path to save checkpoints
    --backbone DSTE \               # Backbone architecture (DSTE)
    --pre-dataset ntu60_2d \        # Pretraining dataset: NTU RGB+D 60 2D keypoints
    --protocol cross_view \         # Evaluation protocol: cross-view split
    --padding zero \                # Padding strategy: zero-padding
    | tee -a ./checkpoint/${dir}/${dir}_pretrain.log  # Save console output to log file

```

