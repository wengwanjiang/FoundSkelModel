
#################################################################################


gpu=3,2,1,0 #$1



dir=ntu2d60_xv_j_dste

mkdir ./checkpoint/${dir} # 0.0005 5e-4
CUDA_VISIBLE_DEVICES="$gpu" python pretrain.py --lr 0.0001   --batch-size 384  --schedule 351 --epochs 451 \
 --moda joint --checkpoint-path ./checkpoint/${dir} --backbone DSTE \
 --pre-dataset ntu60_2d --protocol cross_view --padding zero | tee -a ./checkpoint/${dir}/${dir}_pretrain.log



