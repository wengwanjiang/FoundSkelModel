export GROUND_FOLDER='./scripts/V1_Label/'

gpu=3
detector=$1


# ------------------------------------------------------------------------------ #

dir=ntu60_xs_j_dste
echo "process $dir $detector" | tee -a ./checkpoint/${dir}/${dir}_dete.log
rm -r ./checkpoint/${dir}/detect_each_frame
rm -r ./checkpoint/${dir}/detect_result

CUDA_VISIBLE_DEVICES="${gpu}" python action_detection.py \
  --lr 0.005 --batch-size 512 --backbone STTR --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_0450.pth.tar \
  --finetune-dataset pku_v1 --protocol cross_subject \
  --evaluate $detector | tee -a ./checkpoint/${dir}/${dir}_dete.log

export SOURCE_FOLDER="./checkpoint/${dir}/detect_result/"
python ./scripts/cal_mAP.py | tee -a ./checkpoint/${dir}/${dir}_dete.log

# ------------------------------------------------------------------------------ #

dir=ntu60_xs_j_sttr
echo "process $dir $detector" | tee -a ./checkpoint/${dir}/${dir}_dete.log
rm -r ./checkpoint/${dir}/detect_each_frame
rm -r ./checkpoint/${dir}/detect_result

CUDA_VISIBLE_DEVICES="${gpu}" python action_detection.py \
  --lr 0.005 --batch-size 512 --backbone STTR --moda joint \
  --pretrained  ./checkpoint/${dir}/checkpoint_0450.pth.tar \
  --finetune-dataset pku_v1 --protocol cross_subject \
  --evaluate $detector | tee -a ./checkpoint/${dir}/${dir}_dete.log

export SOURCE_FOLDER="./checkpoint/${dir}/detect_result/"
python ./scripts/cal_mAP.py | tee -a ./checkpoint/${dir}/${dir}_dete.log

