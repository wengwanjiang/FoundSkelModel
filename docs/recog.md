# Skeleton-based Action Recognition (Linear)

In the linear protocol, the backbone network is first pre-trained to learn high-quality skeleton-based motion representations. During evaluation, the backbone parameters are frozen, and only a linear classifier is trained. This setup isolates the representation quality from downstream optimization, providing a fair assessment of the learned features.

In addition, we provide settings for:

- Action Retrieval – Skeleton-based representations are extracted from the pre-trained backbone and compared using a k-nearest neighbors (kNN) search. Retrieval accuracy reflects the semantic consistency and discriminability of the learned feature space.

- Early Action Recognition – This task, also referred to as action prediction, aims to recognize an action as early as possible given only a partially observed sequence. To capture the temporal causality required for this setting, the temporal stream in the transformer employs causal attention operations, ensuring that predictions at any time step are based solely on past observations.

- Semi-Supervised Action Recognition – Only a subset of the training data is used for fine-tuning, testing the model’s generalization capability under limited supervision.

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
