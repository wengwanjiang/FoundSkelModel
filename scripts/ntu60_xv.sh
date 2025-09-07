gpu=1,0 #$1




dir=ntu60_xv_m_sttr
mkdir ./checkpoint/${dir}
CUDA_VISIBLE_DEVICES="$gpu" python pretrain.py --lr 0.0001   --batch-size 356  --schedule 351 --epochs 451 \
 --moda motion --checkpoint-path ./checkpoint/${dir} --backbone STTR \
 --pre-dataset ntu60 --protocol cross_view | tee -a ./checkpoint/${dir}/${dir}_pretrain.log


