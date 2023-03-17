import torch.nn as nn
import torch
from utils import deform_image, spacialGradient_3d, create_meshgrid3d, get_3d_sobel
import reproducing_kernels as rk
from torch.utils.checkpoint import checkpoint
from unet import UNet


class res_block(nn.Module):

    def __init__(self, h, layer=None):
        super().__init__()
        self.conv1 = nn.Conv3d(3, h, 3, bias=False, padding=1)
        self.conv2 = nn.Conv3d(h, h, 3, bias=False, padding=1)
        self.conv3 = nn.Conv3d(h, 1, 3, bias=False, padding=1)

        self.leaky_relu = nn.LeakyReLU()
        self.drop_out = torch.nn.Dropout(p=0.1)

    def forward(self, z, I, J):
        x = torch.cat([z, I, J], dim=1)
        x = self.conv1(x)
        x = self.drop_out(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.drop_out(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        return x


class meta_model(nn.Module):
    """
    Original method, without masking the appearance transformation
    """

    def __init__(self, l, im_shape, device, sigma_v, mu, z0, h=20):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device
        for i in range(l):
            self.res_list.append(res_block(h))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0)

        self.id_grid = create_meshgrid3d(im_shape[2], im_shape[3], im_shape[4], device=device, dtype=torch.HalfTensor)

        self.kernel = rk.GaussianRKHS((sigma_v, sigma_v, sigma_v), border_type='constant')
        self.mu = mu

    def forward(self, source):
        image = []
        image.append(source)
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        self.field = []
        self.grad = []
        for i in range(self.l):
            grad_image = spacialGradient_3d(image[i])
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,4,3,2,1)
            f = self.res_list[i](self.residuals[i], image[i]) * 1/self.l
            self.residuals.append(self.residuals[i] + f)
            deformation = self.id_grid - self.field[i]/self.l
            image.append(deform_image(image[i], deformation) + self.residuals[i+1] * self.mu**2 / self.l)

        return image, self.field, self.residuals, self.grad

class meta_model_local(nn.Module):


    def __init__(self, l, z0, device, sigma_v, mu, h=20):
        super().__init__()
        self.l = l
        self.res_list = []
        im_shape = z0.shape
        self.device = device
        for i in range(l):
            self.res_list.append(res_block(h))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0)
        self.id_grid = create_meshgrid3d(im_shape[2], im_shape[3], im_shape[4], device=device)

        self.kernel = rk.GaussianRKHS((sigma_v, sigma_v, sigma_v), border_type='constant')
        self.mu = mu

    def forward(self, source, source_seg):
        image = []
        image.append(source)
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        self.field = []
        self.grad = []
        mu = self.mu
        for i in range(self.l):
            grad_image = spacialGradient_3d(image[i])
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,4,3,2,1)
            f = self.res_list[i](self.residuals[i], image[i]) * 1 / self.l
            self.residuals.append(self.residuals[i] + f)


            deformation = self.id_grid - self.field[i]/self.l

            source_seg = deform_image(source_seg, deformation)
            image.append(deform_image(image[i], deformation) + self.residuals[i+1] * mu**2 / self.l * source_seg)

        return image, self.field, self.residuals, self.grad

class MetaMorph(nn.Module):
    """
        MetaMorph with local regularizationa and sharp integration scheme
    """

    def __init__(self, l, z0, device, sigma_v, mu, h=50):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device
        for i in range(l):
            self.res_list.append(res_block(h))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0)
        im_shape = z0.shape
        self.id_grid = create_meshgrid3d(im_shape[2], im_shape[3], im_shape[4], device=device)
        self.kernel = rk.GaussianRKHS((sigma_v, sigma_v, sigma_v), border_type='constant')
        self.mu = mu

    def forward(self, source, target, source_seg):
        image = source.clone()
        residuals = [torch.cat([self.z0 for _ in range(source.shape[0])])]
        residuals_deformed = torch.zeros(source.shape).to(self.device)
        field = []
        grad = []
        self.phi = torch.cat([self.id_grid for _ in range(source.shape[0])])
        mu = self.mu
        for i in range(self.l):
            grad_image = spacialGradient_3d(image)
            grad.append(grad_image)
            field.append(self.kernel(-residuals[-1] * grad_image.squeeze(1)))
            field[i] = field[i].permute(0, 4, 3, 2, 1)
            deformation = self.id_grid - field[i] / self.l
            f = checkpoint(self.res_list[i], residuals[-1], image, target) * 1 / self.l
            residuals.append(residuals[-1] + f)
            if i > 0:
                residuals_deformed = deform_image(residuals_deformed, deformation)
            residuals_deformed += residuals[-1]
            self.phi = deform_image(self.phi.permute(0, 4, 3, 2, 1), deformation).permute(0, 4, 3, 2, 1)
            mask = deform_image(source_seg, self.phi)
            image = deform_image(source, self.phi) + residuals_deformed * mu ** 2 / self.l * mask
        self.seg = mask
        return image, field, grad, residuals, residuals_deformed

class shooting_model(nn.Module):
    def __init__(self, l, im_shape, device, sigma_v, mu):
        super().__init__()
        self.l = l
        self.device = device

        self.id_grid = create_meshgrid3d(im_shape[2], im_shape[3], im_shape[4], device=device)
        self.kernel = rk.GaussianRKHS((sigma_v, sigma_v, sigma_v), border_type='constant')
        self.conv1 = nn.Conv3d(3, 1, 3, bias=False, padding=1)
        div_w = get_3d_sobel()
        self.conv1.weight = nn.Parameter(div_w, requires_grad=False)
        self.mu = mu
        self.unet = UNet()

    def forward(self, source, target, source_seg):
        target = torch.cat([target for i in range(source.shape[0])])
        z0 = self.unet(source, source_seg, target)
        return self.shooting(source, z0, source_seg)


    def shooting(self, source, z0, source_seg):
        image = source.clone()
        residuals = [z0]
        residuals_deformed = torch.zeros(source.shape).to(self.device)
        field = []
        grad = []
        self.phi = torch.cat([self.id_grid for _ in range(source.shape[0])])
        mu = self.mu
        for i in range(self.l):
            grad_image = spacialGradient_3d(image)
            grad.append(grad_image)
            field.append(self.kernel(-residuals[-1] * grad_image.squeeze(1)))
            f = self.div(residuals[-1] * field[-1]) / self.l
            residuals.append(residuals[-1] - f)
            field[i] = field[i].permute(0, 4, 3, 2, 1)
            deformation = self.id_grid - field[i] / self.l
            if i > 0:
                residuals_deformed = deform_image(residuals_deformed, deformation)
            residuals_deformed += residuals[-1]
            self.phi = deform_image(self.phi.permute(0, 4, 3, 2, 1), deformation).permute(0, 4, 3, 2, 1)
            mask = deform_image(source_seg, self.phi)
            image = deform_image(source, self.phi) + residuals_deformed * mu ** 2 / self.l * mask
        self.seg = mask
        return image, field, grad, residuals, residuals_deformed

    def div(self, x):
        return self.conv1(x)


class metamorphoses(nn.Module):
    def __init__(self, l, im_shape, device, sigma_v, mu, z0):
        super().__init__()
        self.l = l
        self.device = device

        self.id_grid = create_meshgrid3d(im_shape[2], im_shape[3], im_shape[4], device=device)
        self.kernel = rk.GaussianRKHS((sigma_v, sigma_v, sigma_v), border_type='constant')

        #initialize convolution kernel for divergence
        self.conv1 = nn.Conv3d(3, 1, 3, bias=False, padding=1)
        self.z0 = nn.Parameter(z0)
        div_w = get_3d_sobel()
        self.conv1.weight = nn.Parameter(div_w, requires_grad=False)
        self.mu = mu
    def forward(self, source, target, source_seg):
        return self.shooting(source, source_seg)

    def shooting(self, source, source_seg):
        image = source.clone()
        residuals = [torch.cat([self.z0 for _ in range(source.shape[0])])]
        residuals_deformed = torch.zeros(source.shape).to(self.device)
        field = []
        grad = []
        self.phi = torch.cat([self.id_grid for _ in range(source.shape[0])])
        mu = self.mu
        for i in range(self.l):
            grad_image = spacialGradient_3d(image)
            grad.append(grad_image)
            field.append(self.kernel(-residuals[-1] * grad_image.squeeze(1)))
            f = self.div(residuals[-1] * field[-1]) / self.l
            residuals.append(residuals[-1] - f)
            field[i] = field[i].permute(0, 4, 3, 2, 1)
            deformation = self.id_grid - field[i] / self.l
            if i > 0:
                residuals_deformed = deform_image(residuals_deformed, deformation)
            residuals_deformed += residuals[-1]
            self.phi = deform_image(self.phi.permute(0, 4, 3, 2, 1), deformation).permute(0, 4, 3, 2, 1)
            mask = deform_image(source_seg, self.phi)
            image = deform_image(source, self.phi) + residuals_deformed * mu ** 2 / self.l * mask
        self.seg = mask
        return image, field, grad, residuals, residuals_deformed

    def div(self, x):
        return self.conv1(x)
