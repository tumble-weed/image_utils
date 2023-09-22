import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np
def gaussian_kernel(kernel_size, sigma,channels=1):
    """
    Create a 2D Gaussian kernel with the given size and standard deviation (sigma).
    
    Args:
        kernel_size (int): Size of the kernel (both width and height).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: 2D Gaussian kernel.
    """
    kernel = torch.zeros(1, 1, kernel_size, kernel_size)
    center = kernel_size // 2

    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[0, 0, i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    # kernel = torch.cat([kernel]*channels,dim=1)
    
    return kernel

def apply_gaussian_blur(input_tensor, kernel_size, sigma):
    """
    Apply Gaussian blur to a 4D input tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor to be blurred.
        kernel_size (int): Size of the Gaussian kernel (both width and height).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: Blurred tensor.
    """
    # Create the Gaussian kernel
    channels = input_tensor.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma,channels=channels)
    kernel = kernel.to(input_tensor.device)
    # Ensure that the input tensor has the appropriate number of channels
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Apply convolution with the Gaussian kernel
    padded = F.pad(input_tensor,(kernel_size//2,kernel_size//2,kernel_size//2,kernel_size//2),mode='reflect')
    blurred_tensor = F.conv2d(padded, kernel.tile(channels,1,1,1),groups=channels, padding='valid')

    return blurred_tensor

def resize_with_antialiasing(input_tensor, output_size):
    """
    Resize a PyTorch tensor with anti-aliasing (blurring before downsampling).
    
    Args:
        input_tensor (torch.Tensor): Input tensor to be resized.
        output_size (tuple or int): Desired output size. If a tuple (height, width) is provided, it specifies
            the target size. If an integer is provided, it specifies the target size for both dimensions.
    
    Returns:
        torch.Tensor: Resized tensor with anti-aliasing.
    """
    # Ensure output_size is a tuple
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    # Calculate scaling factors
    h_scale = output_size[0] / input_tensor.size(-2)
    w_scale = output_size[1] / input_tensor.size(-1)

    # Apply Gaussian blur (anti-aliasing)
    if h_scale < 1.0 or w_scale < 1.0:
        # input_tensor = F.gaussian_blur(input_tensor, kernel_size=3, sigma=1.0)
        # kernel_size = 3
        # sigma = 1
        if False:
            sigma = max(h_scale, w_scale)
            kernel_size = int(6 * sigma + 1)
        if True:
            '''
            https://dsp.stackexchange.com/questions/75899/appropriate-gaussian-filter-parameters-when-resizing-image
            '''
            sigma = 1/min(h_scale, w_scale)
            kernel_size = int(6 * sigma + 1)
            # print(kernel_size,sigma)
        input_tensor = apply_gaussian_blur(input_tensor, kernel_size, sigma)
    # import ipdb;ipdb.set_trace()
    
    # Perform resizing with bilinear interpolation
    resized_tensor = F.interpolate(input_tensor, size=output_size, mode='bilinear', align_corners=False)

    return resized_tensor

def masked_conv2d(input, tmp_kernel,mask, padding=0, stride=1):
    if False:
        mask = torch.ones_like(mask)
        print(colored('setting mask to ones in masked_conv2d','red'))        
    # inputu = torch.nn.functional.unfold(input, kernel_size=tmp_kernel.shape[-1], dilation=(1, 1), stride=stride, padding=padding)
    inputu = input.unfold(2,tmp_kernel.shape[-2],1).unfold(3,tmp_kernel.shape[-1],1)
    # masku = torch.nn.functional.unfold(mask, kernel_size=tmp_kernel.shape[-1], dilation=(1, 1), stride=stride, padding=padding)
    masku = mask.unfold(2,tmp_kernel.shape[-2],1).unfold(3,tmp_kernel.shape[-1],1)
    tmp_kernel = tmp_kernel.permute(1,0,2,3)
    # import ipdb;ipdb.set_trace()
    input_dot_kernel = inputu * tmp_kernel[:,:,None,None,:,:]
    mask_dot_kernel =  masku * tmp_kernel[:,:,None,None,:,:]
    masked_input_dot_kernel = (input_dot_kernel * masku)
    denom = mask_dot_kernel.sum(dim=(-1,-2))
    denom = denom + (denom == 0).float()
    out = masked_input_dot_kernel.sum(dim=(-1,-2))/denom
    assert out.shape == input.shape[:2] + ( input.shape[-2] - 2*(tmp_kernel.shape[2]//2),input.shape[-1] - 2*(tmp_kernel.shape[2]//2))
    return out


def masked_filter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    mask:torch.Tensor,
    border_type: str = 'reflect',
    normalized: bool = False,
    padding: str = 'same',
) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2d(input, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input input is not torch.Tensor. Got {type(input)}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Input kernel is not torch.Tensor. Got {type(kernel)}")

    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. Got {type(border_type)}")

    if border_type not in ['constant', 'reflect', 'replicate', 'circular']:
        raise ValueError(
            f"Invalid border type, we expect 'constant', \
        'reflect', 'replicate', 'circular'. Got:{border_type}"
        )

    if not isinstance(padding, str):
        raise TypeError(f"Input padding is not string. Got {type(padding)}")

    if padding not in ['valid', 'same']:
        raise ValueError(f"Invalid padding mode, we expect 'valid' or 'same'. Got: {padding}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if (not len(kernel.shape) == 3) and not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
        raise ValueError(f"Invalid kernel shape, we expect 1xHxW or BxHxW. Got: {kernel.shape}")

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)
    


    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == 'same':
        padding_shape: List[int] = kornia.filters.filter._compute_padding([height, width])
        input = F.pad(input, padding_shape, mode=border_type)
        mask = F.pad(mask, padding_shape, mode=border_type)
    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.


    '''
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    '''
    output = masked_conv2d(input, tmp_kernel, mask, padding=0, stride=1)

    if padding == 'same':
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


