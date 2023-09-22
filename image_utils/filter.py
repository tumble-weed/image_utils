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