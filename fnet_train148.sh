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
ngpu=4
bs=128

ft=True
resume='/mnt/lustre/gongjunchao/IF_WKP/r_00135-era5R32x64-gpus4-batch128-ps1/network-snapshot-training-best.pkl'

loss_name=mse

use_transform=False
insert_frame_aug=False

ps=1


srun -p ai4science --gres=gpu:$ngpu -c 32 -x SH-IDC1-10-140-1-162 bash -c "python train.py \
--gpus=${ngpu} --batch=${bs} --workers=8 \
--finetune=${ft} --resume=${resume} \
--loss_name=${loss_name} \
--use_transform=${use_transform}  --insert_frame_aug=${insert_frame_aug} \
--patch_size=${ps} \
"