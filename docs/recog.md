# Skeleton-based Action Recognition (Linear)

## Scripts

```bash
# ============================================
# Action Recognition - Fine-tuning on NTU60 (Cross-Subject Protocol)
# ============================================

# GPUs to be used
gpu=0,1,2,3  # $1: GPU device IDs

# Checkpoint node identifier (used to select pretrained model file)
node=0450  # $3: checkpoint suffix (e.g., 0450 means epoch 450)

# Experiment directory name
dir=ntu60_xs_j_sttr  # $2: experiment identifier

# ------------------------------------------------------------
# Launch fine-tuning for action recognition
# ------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${gpu}" python action_recognition.py \
    --lr 0.03 \                                             # Learning rate
    --batch-size 512 \                                      # Batch size
    --backbone STTR \                                       # Backbone architecture (STTR / DSTE)
    --moda joint \                                          # Input modality: joint/motion/bone
    --pretrained ./checkpoint/${dir}/checkpoint_${node}.pth.tar \  # Path to pretrained checkpoint
    --finetune-dataset ntu60 \                              # Fine-tuning dataset
    --protocol cross_subject \                              # Evaluation protocol: cross-subject split
    --padding zero \                                        # Padding 2D Skeleton to 3D Skeleton
    --obeserve_ratio 0.9 \                                  # Observation ratio for early-action recog.
    --semi 1 \                                              # Semi-supervised setting     
    | tee -a ./checkpoint/${dir}/${dir}_recog.log           # Save console output to log file


# For other dataset and protocols, just modfiy the --finetune-dataset and --protocol arguments accordingly.
# For additional tasks, e.g., early action recognition and semi-supervised action recognition, and supported options, please refer to args.help() for details.
```
