gpu=0,1 #$1
dir=motion #$2
node=ntu60_xv_moiton_dste #$3
CUDA_VISIBLE_DEVICES="${gpu}" python action_recognition.py \
  --lr 0.01 --batch-size 512 --backbone DSTE --moda motion \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60 --protocol cross_view | tee -a ./checkpoint/${dir}/${dir}_fuck.log
exit 0 
gpu=0,1 #$1
node=0450 #$3
dir=ntu60_xs_j_sttr
CUDA_VISIBLE_DEVICES="${gpu}" python action_recognition.py \
  --lr 0.03 --batch-size 512 --backbone STTR --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_recog.log

# For other dataset and protocols, just modfiy the --finetune-dataset and --protocol arguments
# Notably, Specific --lr should be set for different datasets and protocols

# STTR ntu60 cross_subject  lr = 0.006 
# DSTE ntu60 cross_subject  lr = 0.006 
# DSTE ntu60 cross_view     lr = 0.03
# DSTE ntu120 cross_subject lr = 0.006
# DSTE ntu120 cross_setup   lr = 0.006
# DSTE pku_v2 cross_subject lr = 0.01