# Skeleton-based Detection

In this task, the model is first pre-trained on the NTU60 dataset and then fine-tuned on the PKU-MMD v1 dataset, finetune the entire network. For temporal action detection, we follow a two-stage pipeline that formulates detection as a frame-wise classification problem, with an additional background class to represent non-action frames.

During inference, for long untrimmed videos, we employ a sliding window strategy to sample fixed-length segments, feed each segment into the network, and concatenate the frame-wise predictions along the temporal dimension. A post-processing step then converts these predictions into a set of detection triplets — each consisting of a start frame, an end frame, and an action category — to produce the final detection results.

## Data Preparation

1. Follow the instructions in [docs/pretrain.md](../docs/pretrain.md) to prepare the **PKU-MMD v1** dataset.
2. Convert the dataset into the required detection format as described in the documentation.

## Training

We provide an example for fine-tuning on **PKU-MMD v1 (Cross-Subject Protocol)** using **NTU60 pretrained weights**. The model will be trained for frame-level classification and checkpoints will be saved in `./checkpoints/${dir}`.

```bash
# ============================================
# Skeleton-based Action Detection
# Example: Fine-tuning on PKU-MMD v1 (Cross-Subject Protocol)
# ============================================

# GPUs to be used
gpu=0,1  # $1: GPU device IDs

# Experiment directory name (used to locate pretrained model and save logs)
dir=ntu60_xs_j_sttr  # $2: experiment identifier

# ------------------------------------------------------------
# Launch fine-tuning for action detection
# ------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${gpu}" python action_detection.py \
    --lr 0.005 \                                             # Learning rate
    --batch-size 512 \                                       # Batch size
    --backbone STTR \                                        # Backbone architecture 
    --moda joint \                                           # Input modality
    --pretrained ./checkpoint/${dir}/checkpoint_0450.pth.tar \  # Path to pretrained checkpoint
    --finetune-dataset pku_v1 \                              # Fine-tuning dataset 
    --protocol cross_subject \                               # Evaluation protocol: cross-subject split
    --evaluate None \                                        # Set to None for training
    | tee -a ./checkpoint/${dir}/${dir}_dete.log             # Save console output to log file

```

## Model Checkpoints

- During training, multiple **detector checkpoints** will be saved in:

  ```
  ./checkpoints/${dir}/*pth.tar
  ```

- Since online evaluation of **mAPa** and **mAPv** is not supported in `action_detection.py`, we provide a separate **script scripts/map.sh** for metric computation.

```
export GROUND_FOLDER='./scripts/V1_Label/' 
gpu=3
detector=$1


# ------------------------------------------------------------------------------ #

dir=ntu60_xs_j_dste
echo "process $dir $detector" | tee -a ./checkpoint/${dir}/${dir}_dete.log
rm -r ./checkpoint/${dir}/detect_each_frame
rm -r ./checkpoint/${dir}/detect_result

CUDA_VISIBLE_DEVICES="${gpu}" python action_detection.py \
  --lr 0.005 --batch-size 512 --backbone STTR --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_0450.pth.tar \
  --finetune-dataset pku_v1 --protocol cross_subject \
  --evaluate $detector | tee -a ./checkpoint/${dir}/${dir}_dete.log

export SOURCE_FOLDER="./checkpoint/${dir}/detect_result/"
python ./scripts/cal_mAP.py | tee -a ./checkpoint/${dir}/${dir}_dete.log

# ------------------------------------------------------------------------------ #

dir=ntu60_xs_j_sttr
echo "process $dir $detector" | tee -a ./checkpoint/${dir}/${dir}_dete.log
rm -r ./checkpoint/${dir}/detect_each_frame
rm -r ./checkpoint/${dir}/detect_result

CUDA_VISIBLE_DEVICES="${gpu}" python action_detection.py \
  --lr 0.005 --batch-size 512 --backbone STTR --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_0450.pth.tar \
  --finetune-dataset pku_v1 --protocol cross_subject \
  --evaluate $detector | tee -a ./checkpoint/${dir}/${dir}_dete.log

export SOURCE_FOLDER="./checkpoint/${dir}/detect_result/"
python ./scripts/cal_mAP.py | tee -a ./checkpoint/${dir}/${dir}_dete.log


```



## Evaluation

To evaluate a saved detection model:

1. Specify the checkpoint path using the `--evaluate` argument, for example:

```
--evaluate ./checkpoints/${dir}/detector_1.pth.tar
```

1. Run the evaluation script:

```
bash scripts/map.sh
```

This will compute and report the **mAPa** and **mAPv** scores.

------

## Scripts

All helper scripts are located in the **`scripts/`** directory:

- `scripts/map.sh` – Compute mAPa and mAPv
- `action_detection.py` – Training & evaluation entry point

------

✨ **We encourage you to try out the code, fine-tune on your own datasets, and contribute back!**
