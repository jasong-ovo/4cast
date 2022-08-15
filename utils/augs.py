import numpy as np
import torch

class longtitude_aug(object):
    """
    recurrent shift in longtitude coord
    """
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, feats):
        """
        args:
            feats type(torch.tensor) . T, C, H, W
        """
        T, C, H, W = feats.shape
        bias = int((np.random.uniform(0, 1) * W))
        # print(bias)
        feats = torch.roll(feats, bias, dims=3)
        return feats

class latitude_flip_aug(object):
    """
    flip features along latitude.
    """
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, signals):
        """
        args:
            signals type(torch.tensor). T, C, H, W.
        """
        T, C, H, W = signals.shape
        signals = torch.flip(signals, dims=[2])
        return signals

class identity_aug(object):
    """
    identity map.
    """
    def __init__(self) -> None:
        pass
    
    def __call__(self, signals):
        """
        args:
            signals type(torch.tensor). T, C, H, W.
        """
        return signals

class frame_aug(object):
    """
    insert frames between (t, t+1)
    """
    def __init__(self, p=0.5) -> None:
        self.p = p
        pass

    def __call__(self, signals):
        """
        args:
            feats type(torch.tensor)
        """
        t, c, h, w = signals.shape
        assert t == 3 or t == 4
        cur_p = np.random.uniform(0, 1)
        if cur_p > self.p:
            signals = signals #.to('cuda')
            signals_aug = []
            for i in range(t - 1):
                signal_aug = (signals[i:i+1, :, :, :] + signals[i+1:i+2, :, :, :]) / 2
                signals_aug.append(signal_aug)
            signals_aug = torch.cat(signals_aug, dim=0)
            return signals_aug
        else:
            return signals[:-1, :, :, :]
        


if __name__ == "__main__":
    a = np.array([[[[1, 2, 3], [1, 2, 3]]], [[[4, 5, 6], [1, 2, 3]]]])
    aug = longtitude_aug()
    print(a)
    print(aug(a))