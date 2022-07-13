import torch

def save_model(path_save, mode='training', epoch=0, model=None, optimizer=None, scheduler=None, loss_scaler=None,
               network_kwargs=None, training_set_kwargs=None):
    states = {
        'epoch': epoch,
        'mode': mode,
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'loss_scaler': loss_scaler.state_dict(),
        'network_kwargs': network_kwargs,
        'training_set_kwargs': training_set_kwargs,

    }
    torch.save(states, path_save)

def load_model(path_save, model, optimizer=None, scheduler=None, loss_scaler=None,):
    ckpt = torch.load(path_save, map_location="cpu")
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    loss_scaler.load_state_dict(ckpt['loss_scaler'])
    epoch = ckpt["epoch"]
    return epoch
