
from utils import get_vnorm, get_znorm, save_losses, deform_image, check_diffeo, dice, load_target_seg, inverse_phi
from tqdm.auto import tqdm
import torch

import torch.nn.functional as F
import numpy as np


def train_learning(model, train_loader, test_loader, target, optimizer, config):
    n_iter = len(train_loader)
    l = config["l"]
    mu = config["mu"]
    device = config["device"]
    L2_loss = []
    for e in range(config["n_epoch"]):
        if e == 5 and config["debug"]:
            break
        model.train()
        total_loss_avg = 0
        L2_norm_avg = 0
        residuals_norm_avg = 0
        v_norm_avg = 0

        if config["downsample"]:
            input_target = F.interpolate(target, scale_factor=.5, mode="trilinear")
        else:
            input_target = target

        for i, source in tqdm(enumerate(train_loader)):
            if i == 5 and config["debug"]:
                break
            source_img = source[:, 0]
            source_seg = source[:, 1]
            source_img = source_img.to(device)
            source_seg = source_seg.to(device)
            if config["downsample"]:
                input_source = F.interpolate(source_img, scale_factor=.5, mode="trilinear")
                input_seg = F.interpolate(source_seg, scale_factor=.5, mode="trilinear")
            else:
                input_source = source_img
                input_seg = source_seg
            source_deformed, fields, grad, residuals, residuals_deformed = model(input_source, input_target, input_seg)
            v_norm = get_vnorm(residuals, fields, grad) / config["batch_size"]
            residuals_norm = get_znorm(residuals) / config["batch_size"]

            if config["downsample"]:
                phi_up = F.interpolate(model.phi.permute(0, 4, 3, 2, 1), scale_factor=2, mode="trilinear").permute(0, 4, 3, 2, 1) * 2
                mask_up = F.interpolate(model.seg, scale_factor=2, mode="trilinear")
                residuals = residuals_deformed * mu ** 2
                residuals_up = F.interpolate(residuals, scale_factor=2, mode="trilinear")
                source_deformed = deform_image(source_img, phi_up) + residuals_up / l * mask_up
                phi_inv_small = inverse_phi(fields)
            else:
                phi_inv = inverse_phi(fields)
                phi_inv_small = phi_inv

            L2_norm = ((source_deformed - target) ** 2).sum() / config["batch_size"]
            total_loss = (config["L2_weight"] * L2_norm + config["v_weight"] * v_norm + mu * config["z_weight"] * residuals_norm)

            phi_phi_inv = deform_image(model.phi.permute(0, 4, 3, 2, 1), phi_inv_small).permute(0, 4, 3, 2, 1)
            grid_interp = deform_image(model.id_grid.permute(0, 4, 3, 2, 1), phi_phi_inv).permute(0, 4, 3, 2, 1)
            inv_loss = ((model.id_grid - grid_interp) ** 2).sum()

            total_loss += config['inv_weight'] * inv_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_avg += total_loss / n_iter
            L2_norm_avg += L2_norm / n_iter
            residuals_norm_avg += residuals_norm / n_iter
            v_norm_avg += v_norm / n_iter

        L2_loss.append(L2_norm_avg.detach().cpu())

        print("Training: epoch %d, total loss: %f, L2 norm: %f, v_norm: %f, residuals: %f" % (
            e + 1, total_loss_avg, L2_norm_avg, v_norm_avg, residuals_norm_avg))


        if (e + 1) % config["plot_epoch"] == 0:
            eval(test_loader, target, model, config)
            torch.save(model, config["result_path"] + "/model.pt")


def eval(test_loader, target, model, config):
    model.eval()
    device = config["device"]
    downsample = config["downsample"]

    L2_norm = []
    L2_no_tumor_list = []
    L2_def_list = []
    num_folds = []
    dice_ventricles = []
    target_map = load_target_seg()

    with torch.no_grad():
        for i, source in enumerate(test_loader):
            source_img = source[:, 0].to(device)
            source_seg = source[:, 1].to(device)
            source_map = source[:, 2]
            if downsample:
                source_img_down = F.interpolate(source_img, scale_factor=.5, mode="trilinear")
                target_img_down = F.interpolate(source_img, scale_factor=.5, mode="trilinear")
                source_seg_down = F.interpolate(source_seg, scale_factor=.5, mode="trilinear")
                source_deformed, fields, grad, residuals, residuals_deformed = model(source_img_down, target_img_down, source_seg_down)
                phi_up = F.interpolate(model.phi.permute(0, 4, 3, 2, 1), scale_factor=2, mode="trilinear").permute(0, 4, 3, 2, 1) * 2
                mask_up = F.interpolate(model.seg, scale_factor=2, mode="trilinear")
                residuals = residuals_deformed * model.mu ** 2
                residuals_up = F.interpolate(residuals, scale_factor=2, mode="trilinear")
                source_deformed = deform_image(source_img, phi_up) + residuals_up / model.l * mask_up
                mask_template_space = deform_image(source_seg, phi_up)
                deformed_only = deform_image(source_img, phi_up)
            else:
                source_deformed, fields, grad, residuals, residuals_deformed = model(source_img, target, source_seg)
                mask_template_space = deform_image(source_seg, model.phi)
                deformed_only = deform_image(source_img, model.phi)
                phi_up = model.phi
            num_folds.append(check_diffeo(model.phi.permute(0, 4, 1, 2, 3)).sum().detach().cpu())
            if num_folds[-1] > 0:
                print("the deformation number %d is not diffeomorphic, number of folds %f" % (i, num_folds[-1].item()))
            L2_norm.append(((source_deformed - target) ** 2).sum().detach().cpu())
            L2_no_tumor = ((source_deformed - target) ** 2)[mask_template_space==0].sum().detach().cpu()
            L2_no_tumor_list.append(L2_no_tumor)
            L2_deformed_only = ((deformed_only - target) ** 2).sum().detach().cpu()
            L2_def_list.append(L2_deformed_only)
            deformed_map = deform_image(source_map, phi_up.detach().cpu())
            dice_ventricles.append(dice(deformed_map, target_map))


        print("Validation L2 loss: %f" % (sum(L2_norm) / len(test_loader)),
              "std: %f" % (np.array(L2_norm).std()))
        print("Validation L2 loss no tumor: %f" % (sum(L2_no_tumor_list) / len(test_loader)),
              "std: %f" % (np.array(L2_no_tumor_list).std()))
        print("Validation L2 deformation only: %f" % (sum(L2_def_list) / len(test_loader)),
              "std: %f" % (np.array(L2_def_list).std()))
        print("Average fold number:", sum(num_folds) / len(num_folds), "std: %f" % (np.array(num_folds).std()))
        print("Validation dice:", sum(dice_ventricles) / len(dice_ventricles), "std: %f" % (np.array(dice_ventricles).std()))



