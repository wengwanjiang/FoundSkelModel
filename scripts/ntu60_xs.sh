gpu=0,1 #$1
dir=ntu60_xs_j_sttr #$2
node=45 #$3
CUDA_VISIBLE_DEVICES="${gpu}" python action_recognition.py \
  --lr 0.006 --batch-size 512 --backbone STTR --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_recog.log

dir=ntu60_xs_j_sttr
CUDA_VISIBLE_DEVICES="${gpu}" python action_recognition.py \
  --lr 0.03 --batch-size 512 --backbone STTR --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60


gpu=0,1 #$1
dir=ntu60_xs_b_sttr #$2
node=45 #$3
CUDA_VISIBLE_DEVICES="${gpu}" python action_recognition.py \
  --lr 0.006 --batch-size 512 --backbone STTR --moda bone \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_recog.log

dir=ntu60_xs_b_sttr
CUDA_VISIBLE_DEVICES="${gpu}" python action_recognition.py \
  --lr 0.03 --batch-size 512 --backbone STTR --moda bone \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60

gpu=0,1 #$1
dir=ntu60_xs_m_sttr #$2
node=45 #$3
CUDA_VISIBLE_DEVICES="${gpu}" python action_recognition.py \
  --lr 0.006 --batch-size 512 --backbone STTR --moda motion \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_recog.log

dir=ntu60_xs_m_sttr
CUDA_VISIBLE_DEVICES="${gpu}" python action_recognition.py \
  --lr 0.03 --batch-size 512 --backbone STTR --moda motion \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60
