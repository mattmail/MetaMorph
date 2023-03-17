import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import kornia as K
import numpy as np
from kornia.filters import get_gaussian_kernel1d, filter3d
import nibabel as nib

def deform_image(image, deformation, interpolation="bilinear"):
    _, H, W, D, _ = deformation.shape
    mult = torch.tensor((2 / (H - 1), 2 / (W - 1), 2 / (D - 1)), device=image.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    deformation = deformation * mult - 1
    image = F.grid_sample(image, deformation, interpolation, padding_mode="border", align_corners=True)

    return image.permute(0,1,4,3,2)

def create_meshgrid3d(h,w,d, device="cpu", dtype=torch.HalfTensor):
    d1 = torch.linspace(0, d - 1, d, device=device, dtype=torch.float32)
    d2 = torch.linspace(0, w - 1, w, device=device, dtype=torch.float32)
    d3 = torch.linspace(0, h - 1, h, device=device, dtype=torch.float32)
    meshx, meshy, meshz = torch.meshgrid((d1, d2, d3))
    grid = torch.stack((meshx, meshy, meshz), 3)
    return grid.unsqueeze(0)


def make_grid(size, step):
    grid = torch.zeros(size)
    grid[:,:, ::step] = 1.
    grid[:,:,:, ::step] = 1.
    grid[:, :, :,:, ::step] = 0.
    return grid


def get_vnorm(residuals, fields, grad):
    return torch.stack([(residuals[j] * grad[j].squeeze(1) * (-fields[j].permute(0, 4, 3, 2, 1))).sum() for j in
                              range(len(fields))]).sum()

def get_znorm(residuals):
    return (torch.stack(residuals) ** 2).sum()

def save_losses(L2_loss, L2_val, e, result_path):
    plt.figure()
    x = np.linspace(1, e + 1, e + 1)
    plt.plot(x, L2_loss, color='blue', label="Training")
    plt.plot(x, L2_val, color='red', label="Validation")
    plt.title('L2 norm during training and validation ')
    plt.xlabel('epoch')
    plt.ylabel('L2 norm')
    plt.legend()
    plt.savefig(result_path + '/loss.png')
    plt.clf()

def get_gaussian_kernel3d(size, sigma):
    kernel_x: torch.Tensor = get_gaussian_kernel1d(size, sigma, False)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(size, sigma, False)
    kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    kernel_z: torch.Tensor = get_gaussian_kernel1d(size, sigma, False)
    kernel3d: torch.Tensor  = torch.matmul(kernel_2d.unsqueeze(-1), kernel_z.unsqueeze(-1).t())
    return kernel3d

class GaussianBlur3d(nn.Module):

    def __init__(self, size, sigma):
        super().__init__()
        self.kern = get_gaussian_kernel3d(size, sigma)[None].type(torch.HalfTensor)

    def forward(self, input):
        return filter3d(input, self.kern, "constant", False)

def spacialGradient_3d(image,dx_convention = 'pixel'):
    """

    :param image: Tensor [B,1,H,W,D]
    :param dx_convention:
    :return: Tensor [B,3,H,W,D]

    :Example:
    H,W,D = (50,75,100)
    image = torch.zeros((H,W,D))
    mX,mY,mZ = torch.meshgrid(torch.arange(H),
                              torch.arange(W),
                              torch.arange(D))

    mask_rond = ((mX - H//2)**2 + (mY - W//2)**2).sqrt() < H//4
    mask_carre = (mX > H//4) & (mX < 3*H//4) & (mZ > D//4) & (mZ < 3*D//4)
    mask_diamand = ((mY - W//2).abs() + (mZ - D//2).abs()) < W//4
    mask = mask_rond & mask_carre & mask_diamand
    image[mask] = 1


    grad_image = spacialGradient_3d(image[None,None])
    # grad_image_sum = grad_image.abs().sum(dim=1)
    # iv3d.imshow_3d_slider(grad_image_sum[0])

    """

    # sobel kernel is not implemented for 3D images yet in kornia
    # grad_image = SpatialGradient3d(mode='sobel')(image)
    kernel = get_sobel_kernel_3d().to(image.device).to(image.dtype)
    kernel.requires_grad = False
    spatial_pad = [1,1,1,1,1,1]
    image_padded = F.pad(image,spatial_pad,'replicate').repeat(1,3,1,1,1)
    grad_image =  F.conv3d(image_padded,kernel,padding=0,groups=3,stride=1)
    if dx_convention == '2square':
        _,_,D,H,W, = image.size()
        grad_image[0,0,0] *= (D-1)/2
        grad_image[0,0,1] *= (H-1)/2
        grad_image[0,0,2] *= (W-1)/2

    return grad_image

def get_sobel_kernel_3d():
    return torch.tensor(
    [
        [[[-1,0,1],
          [-2,0,2],
          [-1,0,1]],

         [[-2,0,2],
          [-4,0,4],
          [-2,0,2]],

         [[-1,0,1],
          [-2,0,2],
          [-1,0,1]]],

        [[[-1,-2,-1],
          [0,0,0],
          [1,2,1]],

         [[-2,-4,-2],
          [0,0,0],
          [2,4,2]],

         [[-1,-2,-1],
          [0,0,0],
          [1,2,1]]],

        [[[-1,-2,-1],
          [-2,-4,-2],
          [-1,-2,-1]],

         [[0,0,0],
          [0,0,0],
          [0,0,0]],

         [[1,2,1],
          [2,4,2],
          [1,2,1]]]
    ]).unsqueeze(1)

def check_diffeo(field):
    Jac = K.filters.SpatialGradient3d()(field)
    det = Jac[:, 0, 0] * (Jac[:, 1, 1]*Jac[:,2,2] - Jac[:, 1, 2]*Jac[:,2,1]) - Jac[:,0,1] * (Jac[:, 1, 0]*Jac[:,2,2] -Jac[:,2,0]*Jac[:,1,2]) + Jac[:,0,2] * (Jac[:, 1, 0]*Jac[:,2,1] -Jac[:,2,0]*Jac[:,1,1])
    return det <= 0

def dice(pred, gt):
    eps = 1e-10
    tp = torch.sum(torch.mul(pred, gt))
    fp = torch.sum(torch.mul(pred, 1 - gt))
    fn = torch.sum(torch.mul(1 - pred, gt))
    dice_eps = (2. * tp + eps) / (2. * tp + fp + fn + eps)
    return dice_eps

def load_target_seg():
    seg = nib.load("/home/matthis/datasets/sri_seg.nii.gz").get_fdata().squeeze()[24:-24, 24:-24, 5:-6]
    return torch.tensor(seg[:,::-1].copy()).unsqueeze(0).unsqueeze(0)


def inverse_phi(v):
    device = v[0].device
    id_grid = create_meshgrid3d(v[0].shape[3], v[0].shape[2], v[0].shape[1], device)
    l = len(v)
    phi_inv = id_grid
    for i in range(l):
        deformation = id_grid + v[l-i-1] / l
        phi_inv = deform_image(phi_inv.permute(0, 4, 3, 2, 1), deformation).permute(0, 4, 3, 2, 1)
    return phi_inv


def get_3d_sobel():
    return torch.tensor([
        [[[-1,0,1],
          [-2,0,2],
          [-1,0,1]],

         [[-2,0,2],
          [-4,0,4],
          [-2,0,2]],

         [[-1,0,1],
          [-2,0,2],
          [-1,0,1]]],

        [[[-1,-2,-1],
          [0,0,0],
          [1,2,1]],

         [[-2,-4,-2],
          [0,0,0],
          [2,4,2]],

         [[-1,-2,-1],
          [0,0,0],
          [1,2,1]]],

        [[[-1,-2,-1],
          [-2,-4,-2],
          [-1,-2,-1]],

         [[0,0,0],
          [0,0,0],
          [0,0,0]],

         [[1,2,1],
          [2,4,2],
          [1,2,1]]]
    ]).unsqueeze(0) / 32



