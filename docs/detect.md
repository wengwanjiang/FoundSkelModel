# Skeleton-based Detection

In this task, the model is first pre-trained on the NTU60 dataset and then fine-tuned on the PKU-MMD v1 dataset, finetune the entire network. For temporal action detection, we follow a two-stage pipeline that formulates detection as a frame-wise classification problem, with an additional background class to represent non-action frames.

During inference, for long untrimmed videos, we employ a sliding window strategy to sample fixed-length segments, feed each segment into the network, and concatenate the frame-wise predictions along the temporal dimension. A post-processing step then converts these predictions into a set of detection triplets — each consisting of a start frame, an end frame, and an action category — to produce the final detection results.



## Scripts

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
    --evaluate None \                                        # Evaluation mode: set to None for training
    | tee -a ./checkpoint/${dir}/${dir}_dete.log             # Save console output to log file

```
