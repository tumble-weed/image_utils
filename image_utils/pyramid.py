import torch
TODO = None
from kornia.geometry.transform.pyramid import pyrdown,pyrup

def resize_with_antialiasing():
    pass

def create_pyramid_with_nlevels_kornia(hi_res, 
                                        factor,
                                        pyramid_depth,
                                        border_type = 'reflect', 
                                        align_corners = True, 
                                        ):
    pyramid = [pyrdown(hi_res, 
                        border_type = border_type, 
                        align_corners = align_corners, 
                        factor = factor ** i)for i in range(1,pyramid_depth)]
    pyramid.insert(0,hi_res)
    return pyramid



def create_pyramids(pyramid_scales,reference_images,synthesized_images,mask_with_holes_as_0,mask_selection):
    reference_pyramid = []
    synthesized_pyramid = []
    mask_pyramid = []
    assert ((np.array(pyramid_scales)[1:] - np.array(pyramid_scales)[:1])>=0).all()
    ref_lvl = reference_images
    mask_lvl = mask_with_holes_as_0
    synth_lvl = synthesized_images
    sizes =[]
    for scale in reversed(pyramid_scales):
        '''
        from tv_resize
        If size is an int, smaller edge of the image will be matched to this number. i.e, if height > width, then image will be rescaled to
        (size * height / width, size).
        '''
        if ref_lvl.shape[-1] > ref_lvl.shape[-2]:
            new_size =  scale  , int((ref_lvl.shape[-1] * scale/ref_lvl.shape[-2]))
        else:
            new_size =   int(ref_lvl.shape[-2] * scale/ref_lvl.shape[-1]),scale
        mask_lvl = utils2.resize_with_antialiasing(mask_lvl,new_size) 
        mask_lvl = utils2.binarize_mask(mask_lvl,mask_selection).float()
        ref_lvl = utils2.resize_with_antialiasing(ref_lvl,new_size)
        synth_lvl = utils2.resize_with_antialiasing(synth_lvl,new_size)
        reference_pyramid.append(ref_lvl)
        synthesized_pyramid.append(synth_lvl)
        mask_pyramid.append(mask_lvl)
        sizes.append(new_size)
    mask_pyramid = list(reversed(mask_pyramid))
    reference_pyramid = list(reversed(reference_pyramid))
    synthesized_pyramid = list(reversed(synthesized_pyramid))
    sizes = list(reversed(sizes))
    return reference_pyramid,synthesized_pyramid,mask_pyramid,sizes
