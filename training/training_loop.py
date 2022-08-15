import os
import time
import copy
import pickle
import pynvml
import psutil
import torch
import torch.distributed as dist
import timm
import dnnlib
import numpy as np
from torch_utils import misc
from torch_utils import training_stats
from datasets.dataset import init_dataset
from torch.utils.data.distributed import DistributedSampler
from datasets.sampler import InfiniteSampler, RASampler
from models.module import AFNONet
from utils.io import save_model
from utils.loss_func import get_loss_func

import warnings
import timm.optim
from timm.scheduler import create_scheduler
from torch.nn.parallel import DistributedDataParallel
warnings.filterwarnings("ignore", message="Argument interpolation should be")
warnings.filterwarnings("ignore", message="leaker semaphore objects")
# export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
def get_avaliable_memory(device, rank):
    if rank >= 0:
        # import pdb; pdb.set_trace()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(rank)        # 0表示第一块显卡
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        ava_mem = round(meminfo.free/1024.0**3)

    elif device==torch.device('cpu'):
        mem = psutil.virtual_memory()
        print('current available memory is' +' : '+ str(round(mem.used/1024**3)) +' GIB')
        ava_mem=round(mem.used/1024**3)

    return ava_mem, gpu_util



def training_loop(
        rank                    = 0,        # Rank of the current process in [0, num_gpus[.
        run_dir                 = '.',      # Output directory.
        lr                      = 5e-4,     # learning rate.
        weight_decay            = 0.05,     # weight decay.
        num_gpus                = 8,        # Number of GPUs participating in the training.
        epoch_t                 = 2,       # Epoch of training. 80
        epoch_f                 = 2,       # Epoch of fine-tuning. 50
        resume                  = None,     # Resume from given network pickle
        batch_size              = 64,       # Total batch size for one training iteration
        batch_gpu               = 8,        # Total batch size for each gpu
        random_seed             = 0,        # Global random seed.
        data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
        network_kwargs          = {},       # Options for network.
        training_set_kwargs     =  {},       # Options for training set.
        scheduler_kwargs        =  {},       # Options for scheduler.
        visualize_kwargs        =  {},       # Options for scheduler.
        finetune = False,
        loss_name = 'mse'
):


    accumulation_steps = 1
    loss_step = visualize_kwargs.vis_loss_step
    vis_rank = visualize_kwargs.vis_rank
    dump_epoch = visualize_kwargs.dump_epoch
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    in_chans = network_kwargs.in_chans
    h, w = network_kwargs.img_size


    training_set = init_dataset(training_set_kwargs)
    # training_set_sampler = DistributedSampler(training_set, rank=rank, shuffle=True, num_replicas=num_gpus, seed=random_seed)
    # training_set_sampler = InfiniteSampler(training_set, rank=rank, num_replicas=num_gpus, shuffle=True, seed=random_seed)
    training_set_sampler = RASampler(training_set, rank=rank, num_replicas=num_gpus, shuffle=True, seed=random_seed)
    #
    training_set_dataloader = torch.utils.data.DataLoader(training_set, batch_gpu,  sampler=training_set_sampler, **data_loader_kwargs)

    val_set_kwargs = copy.deepcopy(training_set_kwargs)
    val_set_kwargs.mode = 'valid'
    val_set_kwargs.insert_frame_aug = False
    val_set = init_dataset(val_set_kwargs)
    val_set_sampler = RASampler(val_set, rank=rank, num_replicas=num_gpus, shuffle=True, seed=random_seed)
    val_set_dataloader = torch.utils.data.DataLoader(val_set, batch_gpu, sampler=val_set_sampler, **data_loader_kwargs)

    model = AFNONet(**network_kwargs).to(device).eval()

    if rank == vis_rank:
        z = torch.empty([batch_gpu, network_kwargs.in_chans, *network_kwargs.img_size], device=device)
        with torch.no_grad():
            out = misc.print_module_summary(model, [z,])
        out = None
        z = None
    if rank == vis_rank:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [model]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=vis_rank)          #########################

    model.train()
    criterion = get_loss_func(loss_name, device)#torch.nn.MSELoss()
    param_groups = timm.optim.optim_factory.param_groups_weight_decay(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = torch.cuda.amp.GradScaler(enabled=True)
    lr_scheduler, num_epochs = create_scheduler(scheduler_kwargs, optimizer)
    if rank == vis_rank:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', network_kwargs.img_size)
        print('Num epochs:', num_epochs)
        print()


    ### -------------------------------------pretrain-------------------------------------###
    if rank == vis_rank:

        print("Pretraining start...")
        print('#'*80)
        best_score = 100
    for epoch in range(epoch_t):
        if finetune:
            assert resume is not None
            break
        tick_start_time = time.time()
        losses = []
        model.train()
        for step, batch in enumerate(training_set_dataloader):
            # if step > 3:
            #     break
            # import pdb; pdb.set_trace()
            # continue
            # xs, xt = [x.half().to(device, non_blocking=True) for x in batch]
            batch = batch.half().to(device, non_blocking=True) #batch (B, in_clip+out_clip, C, H, W)
            # import pdb; pdb.set_trace()
            xs, xt = batch[:, 0, :, :, :], batch[:, 1, :, :, :]
            model.requires_grad_(True)
            with torch.cuda.amp.autocast():
                xt_hat = model(xs)  # input: (B, C, H, W)
                loss = criterion(xt_hat, xt)
                loss /= accumulation_steps
            loss_scaler.scale(loss).backward()
            losses.append(loss.item())
            if (step + 1) % accumulation_steps == 0:
                with torch.autograd.profiler.record_function('opt'):
                    params = [param for param in model.parameters() if param.numel() > 0 and param.grad is not None]
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if num_gpus > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= num_gpus
                            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                            grads = flat.split([param.numel() for param in params])
                            for param, grad in zip(params, grads):
                                param.grad = grad.reshape(param.shape)
                loss_scaler.step(optimizer)
                loss_scaler.update()
                optimizer.zero_grad()

            if (step + 1) % loss_step == 0:
                if num_gpus > 1:
                    misc.check_ddp_consistency(model, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                gpumen, gpu_util = get_avaliable_memory(device, rank)
                cpumen = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

                if rank == vis_rank:
                    # print('shape', xs.shape)
                    print("Training Epoch: {:02d} Step: {:04d} cpumen:{:.1f}G gpumen:{:.1f}G gpu_core_util:{:.1f} MSE: {:.4f} ".format(epoch, step + 1, cpumen, gpumen, gpu_util, np.mean(losses[:-(loss_step-1)])))

                # break
            model.requires_grad_(False)

        ## validation ##
        model.eval()
        val_losses = []
        with torch.no_grad():
            for step, batch in enumerate(val_set_dataloader):
                # import pdb; pdb.set_trace()
                # continue
                # xs, xt = [x.half().to(device, non_blocking=True) for x in batch]
                batch = batch.half().to(device, non_blocking=True) #batch (B, in_clip+out_clip, C, H, W)
                xs, xt = batch[:, 0, :, :, :], batch[:, 1, :, :, :]
                with torch.cuda.amp.autocast():
                    xt_hat = model(xs)
                    loss = criterion(xt_hat, xt)
                    loss /= accumulation_steps
                    if num_gpus > 1:
                        dist.all_reduce(loss)
                        loss /= num_gpus
                        # print(loss)
                val_losses.append(loss.item())
                
        tick_end_time = time.time()
        tick_time = tick_end_time - tick_start_time
        if num_gpus > 1:
            misc.check_ddp_consistency(model, ignore_regex=r'.*\.[^.]+_(avg|ema)')
        gpumen, gpu_util = get_avaliable_memory(device, rank)
        cpumen = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

        if rank == vis_rank:
            print("Training Epoch: {:02d} cpumen:{:.1f}G gpumen:{:.1f}G gpu_core_util:{:.1f}% MSE:{:.4f} valMSE:{:.4f} Time: {:02d}min".format(epoch, cpumen, gpumen, gpu_util, np.mean(losses), np.mean(val_losses), int(tick_time //60)))
            if best_score > np.mean(val_losses):
                best_score = np.mean(val_losses)
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs), network_kwargs=dict(network_kwargs))
                # print("######################")
                for name, module in [('model', model)]:
                    print(name)
                    if module is not None:
                        if num_gpus > 1:
                            pass
                            # print(module)
                            # misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                            # print("out")
                        state = copy.deepcopy(module).eval().requires_grad_(False).cpu().state_dict()
                    snapshot_data[name] = state
                    del module  # conserve memory
                # print("############out##################")
                snapshot_pkl = os.path.join(run_dir, 'network-snapshot-training-best.pkl')
                resume_pkl = snapshot_pkl
                torch.save(snapshot_data, snapshot_pkl)

        lr_scheduler.step(epoch)
        # if (epoch+1) % dump_epoch == 0:




    model.eval()
    # Save network snapshot for training.
    snapshot_pkl = None
    snapshot_data = None
    resume_pkl = resume
    if (resume is None or epoch_t > 0) and not finetune:
        snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs), network_kwargs=dict(network_kwargs))
        for name, module in [('model', model)]:
            if module is not None:
                if num_gpus > 1:
                    misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                state = copy.deepcopy(module).eval().requires_grad_(False).cpu().state_dict()
            snapshot_data[name] = state
            del module  # conserve memory
        snapshot_pkl = os.path.join(run_dir, 'network-snapshot-training-epoch-{:02d}.pkl'.format(epoch_t))
        if rank == vis_rank:
            torch.save(snapshot_data, snapshot_pkl)            ###################
        # resume_pkl = snapshot_pkl
        resume_pkl = os.path.join(run_dir, 'network-snapshot-training-best.pkl')
        pass
    else:
        resume_pkl = resume
        
    param_groups = None
    optimizer = None
    loss_scaler = None
    lr_scheduler = None
    training_set = None
    training_set_sampler = None
    training_set_dataloader = None
    snapshot_pkl = None
    snapshot_data = None
    model = None

    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    ### *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ###
    ### -------------------------------------finetuning-------------------------------------###
    if rank == vis_rank:
        print()
        print("Finetuning start...")
        print('#' * 80)
    model = AFNONet(**network_kwargs).to(device).eval()
    # resume_pkl = "/mnt/lustre/gongjunchao/IF_WKP/00122-era5R32x64-gpus4-batch128/network-snapshot-training-best.pkl"
    if (resume_pkl is not None) and (rank == vis_rank):
        print(f'Resuming from "{resume_pkl}"')

        resume_data = torch.load(resume_pkl, map_location=device)
        model.load_state_dict(resume_data['model'])

        # for name, module in [('model', model)]:
        #     misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    if rank == vis_rank:
        print(f'Distributing across {num_gpus} GPUs...')
        best_score = 100
    for module in [model]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=vis_rank)

    model.train()
    criterion = get_loss_func(loss_name, device)#torch.nn.MSELoss()
    finetune_scheduler_kwargs = copy.deepcopy(scheduler_kwargs)
    finetune_scheduler_kwargs.epochs = max(epoch_f - finetune_scheduler_kwargs.warmup_epochs, 1)
    if num_gpus > 1:
        misc.check_ddp_consistency(model, ignore_regex=r'.*\.[^.]+_(avg|ema)')

    param_groups = timm.optim.optim_factory.param_groups_weight_decay(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = torch.cuda.amp.GradScaler(enabled=True)
    lr_scheduler, num_epochs = create_scheduler(finetune_scheduler_kwargs, optimizer)
    finetuning_set_kwargs = copy.deepcopy(training_set_kwargs)
    finetuning_set_kwargs.length = 2
    finetuning_set = init_dataset(finetuning_set_kwargs)
    finetuning_set_sampler = RASampler(finetuning_set, rank=rank, num_replicas=num_gpus, shuffle=True, seed=random_seed)
    finetuning_set_dataloader = torch.utils.data.DataLoader(finetuning_set, batch_gpu, sampler=finetuning_set_sampler,
                                                          **data_loader_kwargs)

    finetuning_val_set_kwargs = copy.deepcopy(finetuning_set_kwargs)
    finetuning_val_set_kwargs.insert_frame_aug = False
    finetuning_val_set_kwargs.mode = 'valid'
    finetuning_val_set = init_dataset(finetuning_val_set_kwargs)
    finetuning_val_set_sampler = RASampler(finetuning_val_set, rank=rank, num_replicas=num_gpus, shuffle=True, seed=random_seed)
    finetuning_val_set_dataloader = torch.utils.data.DataLoader(finetuning_val_set, batch_gpu, sampler=finetuning_val_set_sampler,
                                                          **data_loader_kwargs)

    for epoch in range(epoch_f):
        tick_start_time = time.time()
        losses = []
        model.train()
        for step, batch in enumerate(finetuning_set_dataloader):
            # xs, xm, xt = [x.half().to(device, non_blocking=True) for x in batch]
            batch = batch.half().to(device, non_blocking=True)
            xs, xm, xt = batch[:, 0, :, :, :], batch[:, 1, :, :, :], batch[:, 2, :, :, :]
            model.requires_grad_(True)
            with torch.cuda.amp.autocast():
                xm_hat = model(xs)
                xt_hat = model(xm_hat)
                loss_m = criterion(xm_hat, xm)
                loss_t = criterion(xt_hat, xt)
                loss = loss_m + loss_t
                loss /= accumulation_steps
            loss_scaler.scale(loss).backward()
            losses.append(loss.item())
            if (step + 1) % accumulation_steps == 0:
                with torch.autograd.profiler.record_function('opt'):
                    params = [param for param in model.parameters() if param.numel() > 0 and param.grad is not None]
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if num_gpus > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= num_gpus
                            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                            grads = flat.split([param.numel() for param in params])
                            for param, grad in zip(params, grads):
                                param.grad = grad.reshape(param.shape)
                loss_scaler.step(optimizer)
                loss_scaler.update()
                optimizer.zero_grad()

            if (step + 1) % loss_step == 0:
                if num_gpus > 1:
                    misc.check_ddp_consistency(model, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                gpumen, gpu_util = get_avaliable_memory(device, rank)
                cpumen = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

                if rank == vis_rank:
                    print("Finetuning Epoch: {:02d} Step: {:04d} cpumen:{:.1f}G gpumen:{:.1f}G gpu_core_util:{:.1f}% MSE: {:.4f} ".format(epoch, step + 1, cpumen, gpumen, gpu_util, np.mean(losses[:-(loss_step-1)])))
                # break
            model.requires_grad_(False)
        
        ## validation ##
        model.eval()
        val_losses = []
        with torch.no_grad():
            for step, batch in enumerate(finetuning_val_set_dataloader):
                # import pdb; pdb.set_trace()
                # continue
                # xs, xt = [x.half().to(device, non_blocking=True) for x in batch]
                batch = batch.half().to(device, non_blocking=True)
                xs, xm, xt = batch[:, 0, :, :, :], batch[:, 1, :, :, :], batch[:, 2, :, :, :]#batch (B, in_clip+out_clip, C, H, W)
                with torch.cuda.amp.autocast():
                    xm_hat = model(xs)
                    xt_hat = model(xm_hat)
                    loss_m = criterion(xm_hat, xm)
                    loss_t = criterion(xt_hat, xt)
                    loss = loss_m + loss_t
                    loss /= accumulation_steps
                val_losses.append(loss.item())
        

        tick_end_time = time.time()
        tick_time = tick_end_time - tick_start_time
        if num_gpus > 1:
            misc.check_ddp_consistency(model, ignore_regex=r'.*\.[^.]+_(avg|ema)')
        gpumen, gpu_util = get_avaliable_memory(device, rank)
        cpumen = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        if rank == vis_rank:
            print("Finetuning Epoch: {:02d} lr:{:.8f} cpumen:{:.1f}G gpumen:{:.1f}G gpu_core_util:{:.1f}% MSE:{:.4f} valMSE:{:.4f} Time: {:02d}min".format(epoch, optimizer.param_groups[0]['lr'], cpumen, gpumen, gpu_util, np.mean(losses), np.mean(val_losses), int(tick_time //60)))
            if best_score > np.mean(val_losses):
                best_score = np.mean(val_losses)
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs), network_kwargs=dict(network_kwargs))
                for name, module in [('model', model)]:
                    if module is not None:
                        if num_gpus > 1:
                            pass
                            # misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        state = copy.deepcopy(module).eval().requires_grad_(False).cpu().state_dict()
                    snapshot_data[name] = state
                    del module  # conserve memory
                snapshot_pkl = os.path.join(run_dir, 'network-snapshot-finetuning-best.pkl')
                resume_pkl = snapshot_pkl
                torch.save(snapshot_data, snapshot_pkl)
        lr_scheduler.step(epoch)

    # Save network snapshot for training.
    snapshot_pkl = None
    snapshot_data = None
    snapshot_data = dict(training_set_kwargs=dict(finetuning_set_kwargs), network_kwargs=dict(network_kwargs))
    for name, module in [('model', model)]:
        if module is not None:
            if num_gpus > 1:
                misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
            state = copy.deepcopy(module).eval().requires_grad_(False).cpu().state_dict()
        snapshot_data[name] = state
        del module  # conserve memory
    snapshot_pkl = os.path.join(run_dir, 'network-snapshot-finetuning-epoch-{:02d}.pkl'.format(epoch_t))
    if rank == vis_rank:
        torch.save(snapshot_data, snapshot_pkl)
    resume_pkl = snapshot_pkl
    snapshot_pkl = None
    snapshot_data = None

    return None



