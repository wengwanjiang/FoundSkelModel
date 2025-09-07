gpu=0,1 # $1
dir=ntu60_xs_j_sttr # $2


CUDA_VISIBLE_DEVICES="${gpu}" python action_detection.py \
  --lr 0.005 --batch-size 512 --backbone STTR --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_0450.pth.tar \
  --finetune-dataset pku_v1 --protocol cross_subject \
  --evaluate None | tee -a ./checkpoint/${dir}/${dir}_dete.log

dir=ntu60_xs_j_dste
CUDA_VISIBLE_DEVICES="${gpu}" python action_detection.py \
  --lr 0.005 --batch-size 512 --backbone DSTE --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_0450.pth.tar \
  --finetune-dataset pku_v1 --protocol cross_subject \
  --evaluate None | tee -a ./checkpoint/${dir}/${dir}_dete.log
