import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import deform_image, get_contours, grayscale_to_rgb, overlay_contours, load_target_seg, dice, check_diffeo
import numpy as np

import cv2



from prepare_data import load_brats_2021
result_path="/home/matthis/Nextcloud/3D_metamorphoses/results/"
#result_path="/home/infres/maillard/3D_metamorphoses/results/"
model_path = "meta_model_1024_1322/model.pt" #inv_loss = 0.0001
#model_path = "meta_model_1024_1648/model.pt" #inv_loss = 0.0001 , sigma=10
#model_path = "meta_model_1020_1003/model.pt" #inv_loss = 0.005
#model_path = "meta_model_0530_1148/model.pt" #results l4e-4
#model_path = "meta_model_1025_1716/model.pt" #h=50
#model_path = "meta_model_1108_1306/model.pt"
# model_path = "meta_model_1115_0925/model.pt" #this is a good one
#model_path = "meta_model_1118_1630/model.pt"
#model_path = "meta_model_1124_1347/model.pt"
model_path = "meta_model_1122_0939/model.pt"
#model_path = "meta_model_1124_1559/model.pt"

device = "cuda:0"
model = torch.load(result_path + model_path, map_location=device)
model.device = device


"""fig, ax = plt.subplots()
ax.axis("off")
im = ax.imshow(model.z0[:, :, :, :, 80].squeeze().detach().cpu().t() * model.mu**2/model.l, cmap="coolwarm")
fig.colorbar(im)
fig.tight_layout()
plt.show()"""
"""model.phi = []
model.residuals = []
model.residuals_deformed = []
model.field = []
model.grad = []
torch.save(model, result_path + "meta_model_0530_1145/model_light.pt")"""

torch.manual_seed(5)
target_map = load_target_seg().float()

#list_files = ['00314', '01312', '01393', '00339', '00263']
list_files = ['01261', '00494', '01340', '01399', '01104', '01312', '01452']
list_files = ['01399']
list_files = ['00131']

train_loader, test_loader, target_img, _ = load_brats_2021(device, 1, get_ventricles=True, return_name=True)
target = target_img.to(device)
model.eval()
L2_list = []
L2_no_tumor = []
L2_def_only = []
dice_list = []
best_dice = 0
fold_list = []
with torch.no_grad():
    model.train()
    for i, (source, name) in enumerate(train_loader):
        if name[0].split("_")[1] in list_files or True:
            slice=75
            source_img = source[:, 0].to(device)
            source_seg = source[:, 1].to(device)
            #source_map = source[:, 2]
            phi_list = []
            deform_list = []
            for j in range(20):
                print(j)
                source_deformed, fields, grad, _, _ = model(source_img, target, source_seg)
                phi_list.append(model.phi.detach().cpu())
                deform_list.append(source_deformed.detach().cpu())

            deform_list = torch.stack(deform_list)
            phi_list = torch.stack(phi_list)
            uc_def = torch.std(deform_list, dim=0)
            uc_phi = torch.std(phi_list, dim=0)
            avg_def = torch.mean(deform_list, dim=0)
            avg_phi = torch.mean(phi_list, dim=0)
            deform_only = deform_image(source_img.detach().cpu(), avg_phi).squeeze()
            #uc_phi = uc_phi.sum(dim=4).permute(0,3,2,1)
            print(uc_phi.shape)
            fig, ax = plt.subplots(2, 3)
            ax[0, 1].imshow(avg_def.squeeze()[:,:,slice].t(), cmap="gray", vmin=0, vmax=1)
            ax[0, 1].set_title("mean transformation")
            ax[0, 2].imshow(uc_def.squeeze()[:,:,slice].t(), cmap="gray")
            ax[0, 2].set_title("uc meta.")
            ax[1, 1].imshow(deform_only.squeeze()[:,:,slice].t(), cmap="gray", vmin=0, vmax=1)
            ax[1, 1].set_title("mean def.")
            ax[1, 2].imshow(uc_phi.squeeze().permute(1,2,0,3)[:,:,slice])
            ax[1, 2].set_title("uc deformation")
            ax[0, 0].imshow(source_img.squeeze().detach().cpu()[:,:,slice].t(), cmap="gray", vmin=0, vmax=1)
            ax[0, 0].set_title("source")
            ax[1, 0].imshow(target.squeeze().detach().cpu()[:,:,slice].t(), cmap="gray", vmin=0, vmax=1)
            ax[1, 0].set_title("target")
            ax[1, 1].axis("off")
            ax[0, 1].axis("off")
            ax[1, 0].axis("off")
            ax[0, 0].axis("off")
            ax[0, 2].axis("off")
            ax[1, 2].axis("off")
            # plt.title("image %d" %i)
            plt.subplots_adjust(wspace=0, hspace=0.01)
            plt.margins(0, 0)
            #plt.savefig("../results/figs/uncertainty.png")
            plt.show()
            #source_deformed, fields, grad, residuals, residuals_deformed = model(source_img, target, source_seg)
            #mask_template_space = deform_image(source_seg, model.phi)
            #deformed_only = deform_image(source_img, model.phi)

            """L2_list.append(((source_deformed - target) ** 2).sum().detach().cpu().item())
            L2_no_tumor.append(((1-mask_template_space) * (source_deformed - target) ** 2).sum().detach().cpu().item())
            L2_def_only.append(((deformed_only - target) ** 2).sum().detach().cpu().item())
            #deformed_label = deform_image(source_map.to(device), model.phi)
            fold_list.append(check_diffeo(model.phi.detach().cpu().permute(0,4,1,2,3)).sum())"""

            """d_score = dice(deformed_label.detach().cpu(), target_map)
            dice_list.append(d_score)
            if d_score > best_dice:
                best_dice = d_score
                best_image = name[0].split("_")[1]"""

            save = False

            plt.figure()
            plot_image = grayscale_to_rgb(source_img[:,:,:,:,slice].detach().cpu())
            contours = get_contours(source_seg[:,:,:,:,slice].detach().cpu())
            plot_image = overlay_contours(plot_image, contours, "blue")
            plt.imshow(plot_image.squeeze().permute(1,0,2))
            plt.axis('off')
            if save:
                plt.savefig("../results/images/source_" + name[0].split("_")[1]+".png", bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.figure()
            plt.imshow(source_deformed.squeeze().detach().cpu()[:,:,slice].permute(1,0), cmap='gray', vmin=0, vmax=1)
            plt.show()

            """plt.figure()
            plot_image = grayscale_to_rgb(deformed_only[:, :, :, :, slice].detach().cpu())
            contours = get_contours(target_map[:, :, :, :, slice].detach().cpu())
            plot_image = overlay_contours(plot_image, contours)
            plt.imshow(plot_image.squeeze().permute(1, 0, 2))
            plt.axis('off')
            if save:
                plt.savefig("../results/images/deformation_" + name[0].split("_")[1] + ".png", bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.figure()

            plot_image = grayscale_to_rgb(source_deformed[:, :, :, :, slice].detach().cpu())
            contours = get_contours(target_map[:, :, :, :, slice].detach().cpu() )
            plot_image = overlay_contours(plot_image, contours)
            plt.imshow(plot_image.squeeze().permute(1, 0, 2))
            plt.axis('off')
            if save:
                plt.savefig("../results/images/meta_" + name[0].split("_")[1] + ".png", bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.figure()

            plt.imshow(target_img[:,:,:,:,slice].squeeze().detach().cpu().t(), cmap='gray', vmin=0., vmax=1.)
            plt.axis('off')
            if save:
                plt.savefig("../results/images/target_" + name[0].split("_")[1] + ".png", bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.figure()"""

            """fig, ax = plt.subplots(2, 3)
            ax = ax.ravel()
            plot_image = grayscale_to_rgb(source_img[:, :, :, :, slice].detach().cpu())
            contours = get_contours(source_map[:, :, :, :, slice].detach().cpu())
            plot_image = overlay_contours(plot_image, contours)
            ax[0].imshow(plot_image.squeeze().permute(1, 0, 2), vmin=0, vmax=1, cmap="gray")
            ax[0].set_title("Source")
            ax[1].imshow(target[:, :, :, :, slice].squeeze().detach().cpu().t(), vmin=0, vmax=1, cmap="gray")
            ax[1].set_title("Target")
            plot_image = grayscale_to_rgb(source_deformed[:, :, :, :, slice].detach().cpu())
            contours = get_contours(target_map[:, :, :, :, slice].detach().cpu())
            plot_image = overlay_contours(plot_image, contours)
            #contours = get_contours(1. * (deformed_label[:, :, :, :, slice].detach().cpu() > 0.5))
            #plot_image = overlay_contours(plot_image, contours, "green")
            ax[3].imshow(plot_image.squeeze().permute(1, 0, 2))
            ax[3].set_title("Meta")
            plot_image = grayscale_to_rgb(deformed_only[:, :, :, :, slice].detach().cpu())
            contours = get_contours(target_map[:, :, :, :, slice].detach().cpu())
            plot_image = overlay_contours(plot_image, contours)"""
            """contours = get_contours(1.*(deformed_label[:, :, :, :, slice].detach().cpu() > 0.5))
            plot_image = overlay_contours(plot_image, contours, "green")"""
            """ax[2].imshow(plot_image.squeeze().permute(1, 0, 2))
            ax[2].set_title("deformed")
            ax[4].imshow(source_seg[:, :, :, :, slice].squeeze().detach().cpu().t(), vmin=0, vmax=1, cmap="gray")
            ax[4].set_title("deformed")
            ax[5].imshow(mask_template_space[:, :, :, :, slice].squeeze().detach().cpu().t(), vmin=0, vmax=1, cmap="gray")
            ax[5].set_title("deformed")
            ax[0].axis("off")
            ax[1].axis("off")
            ax[2].axis("off")
            ax[3].axis("off")
            ax[4].axis("off")
            ax[5].axis("off")
            fig.tight_layout()
            fig.suptitle("Image: %s" %name[0].split("_")[1] + ",   dice: " + str(d_score.item()))
            plt.show()"""
    print("Best image:", best_image, ", dice score:", best_dice)
    print("L2 average", np.mean(np.array(L2_list)), "std:", np.std(np.array(L2_list)))
    print("dice", np.mean(np.array(dice_list)), "std:", np.std(np.array(dice_list)))
    print("L2 outside average", np.mean(np.array(L2_no_tumor)), "std:", np.std(np.array(L2_no_tumor)))
    print("L2 def only", np.mean(np.array(L2_def_only)), "std:", np.std(np.array(L2_def_only)))
    print("folds", np.mean(np.array(fold_list)), "std:", np.std(np.array(fold_list)))