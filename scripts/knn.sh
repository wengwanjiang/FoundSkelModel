gpu=0 #$1
dir=ntu120_xe_j_dste #$2
node=0450 #$3
CUDA_VISIBLE_DEVICES="${gpu}" python action_retrieval.py \
  --lr 0.006 --batch-size 512 --backbone DSTE --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu120 --protocol cross_setup | tee -a ./checkpoint/${dir}/${dir}_knn.log
exit 0 
dir=ntu60_xs_j_sttr
CUDA_VISIBLE_DEVICES="${gpu}" python action_retrieval.py \
  --lr 0.03 --batch-size 512 --backbone STTR --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_${node}.pth.tar \
  --finetune-dataset ntu60 --protocol cross_subject | tee -a ./checkpoint/${dir}/${dir}_knn.log
  
# For other dataset and protocols, just modfiy the --finetune-dataset and --protocol arguments