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
bs=32

ft=False
resume=None

loss_name=mse

use_transform=False
insert_frame_aug=False
std_trans=False

ps=8


srun -p ai4science --gres=gpu:$ngpu -c 24 -x SH-IDC1-10-140-1-162 bash -c "python train.py \
--gpus=${ngpu} --batch=${bs} --workers=8 \
--finetune=${ft} --resume=${resume} \
--loss_name=${loss_name} \
--use_transform=${use_transform}  --insert_frame_aug=${insert_frame_aug} --std_trans=${std_trans} \
--patch_size=${ps} \
"