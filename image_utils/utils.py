
import torch
import numpy as np

def extract_patches_6d(img,patch_size,stride):
    return img.unfold(2,patch_size,stride).unfold(3,patch_size,stride)
def extract_patches_5d(img,patch_size,stride):
    patches_6d = extract_patches_6d(img,patch_size,stride)
    patches_5d = patches_6d.view(*img.shape[:2],-1,patch_size,patch_size)
    return patches_5d


def binarize_mask(mask,mode,other=None):
    if mode == 'equal_to_1':
        return torch.isclose(mask,torch.ones_like(mask))
    elif mode == 'all':
        return torch.ones_like(mask).bool()
    elif mode == 'larger_than_0.5':
        return (mask>=0.5)
    elif mode == 'smaller_than_0.5':
        return (mask<0.5)    
    elif mode == 'equal_to_0':    
        return torch.isclose(mask,torch.zeros_like(mask))
    elif mode == 'complement':
        assert other is not None
        return ~other
    else:
        assert False

