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
resume=/mnt/lustre/gongjunchao/IF_WKP/00158-era5R32x64-gpus1-batch32/network-snapshot-finetuning-best.pkl

loss_name=mse

use_transform=False
insert_frame_aug=False
std_trans=False

ps=8

# srun -p ai4science --gres=gpu:8 -c 256 bash -c "python train.py --gpus=8 --batch=56" 
#--resh 720 --resw 1440 --dataset ERA5CephDataset

srun -p ai4science --gres=gpu:$ngpu -c 24 -x SH-IDC1-10-140-1-162 bash -c "python evaluate.py \
--gpus=${ngpu} --batch=${bs} --workers=8 \
--resume=${resume} \
--patch_size=${ps} \
--std_trans=${std_trans} \
"