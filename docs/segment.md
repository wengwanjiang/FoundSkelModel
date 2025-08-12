# Skeleton-based Segmentation

In this task, the model is first pre-trained on the NTU60 dataset and then fine-tuned on the PKU-MMD v2 dataset, updating the entire network. Skeleton-based action segmentation aims to predict the action category for each frame in a sequence, thereby producing a fine-grained temporal labeling of actions.

We adopt the end-to-end pipeline proposed in [DeST](https://github.com/HaoyuJi/LaSA), which directly feeds the entire long sequence into the network without splitting it into short clips. This design allows the model to leverage full-sequence temporal context, enabling accurate frame-wise predictions without the need for a separate post-processing stage.

## Scripts

```bash
# ============================================
# Skeleton-based Action Segmentation
# Example: Fine-tuning on PKU-MMD v2 (Cross-Subject Protocol)
# ============================================

# GPUs to be used
gpu=0,1  # $1: GPU device IDs

# Experiment directory name (used to locate pretrained model and save logs)
dir=ntu60_xs_j_sttr  # $2: experiment identifier

# ------------------------------------------------------------
# Launch fine-tuning for action segmentation
# ------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${gpu}" python action_segmentation.py \
    --lr 0.005 \                                             # Learning rate
    --batch-size 512 \                                       # Batch size
    --backbone STTR \                                        # Backbone architecture 
    --moda joint \                                           # Input modality
    --pretrained ./checkpoint/${dir}/checkpoint_0450.pth.tar \  # Path to pretrained checkpoint
    --finetune-dataset pku_v2 \                              # Fine-tuning dataset 
    --protocol cross_subject \                               # Evaluation protocol: cross-subject split
    --evaluate None \                                        # Evaluation mode: set to None for training
    | tee -a ./checkpoint/${dir}/${dir}_seg.log             # Save console output to log file

```
