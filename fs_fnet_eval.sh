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

# srun -p ai4science --gres=gpu:8 -c 256 bash -c "python train.py --gpus=8 --batch=56" 
#--resh 720 --resw 1440 --dataset ERA5CephDataset

srun -p ai4science --gres=gpu:$ngpu -c 32 -x SH-IDC1-10-140-1-115 bash -c "python evaluate_fs.py --gpus=${ngpu} --batch-size=64 --workers=8"