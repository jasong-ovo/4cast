while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

tag=fnet
ngpu=1
bs=64
#test new dataset
ft=False
resume=None

loss_name=mse

use_longtitude_aug=False
insert_frame_aug=False
use_latitude_aug=True

ps=8


srun -p ai4science --gres=gpu:$ngpu -c 24 -x SH-IDC1-10-140-1-162 bash -c "python train.py \
--gpus=${ngpu} --batch=${bs} --workers=8 \
--finetune=${ft} --resume=${resume} \
--loss_name=${loss_name} \
--use_longtitude_aug=${use_longtitude_aug}  --insert_frame_aug=${insert_frame_aug} --use_latitude_aug=${use_latitude_aug} \
--patch_size=${ps} \
"