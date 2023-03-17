import numpy as np
import torch
import nibabel as nib
import os
import random
import time
from torch.utils.data import Dataset, DataLoader

def load_brats_2021(device, batch_size, infres=False, return_name=False):
    if infres:
        root = "/home/infres/maillard/"
    else:
        root = "/home/matthis/Nextcloud/"
    config = dict()
    config["root"] = root
    config["result_path"] = root + "3D_metamorphoses/results/" + "meta_model_" + time.strftime("%m%d_%H%M", time.localtime())
    if not os.path.exists(config["result_path"]):
        os.mkdir(config["result_path"])
    print(config["result_path"])
    target_img = nib.load(root + "templates/T1_brain.nii").get_fdata().squeeze()
    target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())
    target_img = target_img[:, ::-1, 5:-6]
    target_img = target_img[24:-24, 24:-24]
    target_img = torch.from_numpy(target_img.copy()).unsqueeze(0).unsqueeze(0).to(device).float()
    train_set = BratsDataset(infres=infres, return_name=return_name)
    test_set = BratsDataset(test_paths=train_set.test_paths, get_ventricles=True, infres=infres, return_name=return_name)
    train_load = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=32)
    test_load = DataLoader(test_set)
    print("Number of train files:", len(train_load))
    print("Number of test files:", len(test_load))
    return train_load, test_load, target_img, config


class BratsDataset(Dataset):
    def __init__(self, test_file="3D_metamorphoses/src/test_split.txt", test_paths=None, get_ventricles=False, infres=False, return_name=False):
        super(BratsDataset, self).__init__()
        self.get_ventricles = get_ventricles
        if infres:
            root = "/home/infres/maillard/"
            root_test = root
        else:
            root = "/home/matthis/datasets/"
            root_test = "/home/matthis/Nextcloud/"
        if test_paths is None:
            data_dir = root + "BraTS2021_preproc/"
            textfile = open(root_test + test_file, "r")
            i = 0
            self.img_path = []
            self.test_paths = []
            if test_file is None:
                for file in os.listdir(data_dir):
                    if file[:5] == "BraTS":
                        image = os.path.join(data_dir, file, file + "_t1.npy")
                        seg = os.path.join(data_dir, file, file + "_seg.npy")
                        if np.random.rand() > 0.2:
                            self.img_path.append([image, seg])
                        else:
                            self.test_paths.append([image, seg])
                            textfile.write(file + "\n")
            else:
                self.test_paths = [(os.path.join(data_dir, file[:-1], file[:-1] + "_t1.npy"), os.path.join(data_dir, file[:-1], file[:-1] + "_seg.npy")) for file in textfile.readlines()]
                for file in os.listdir(data_dir):
                    if file[:5] == "BraTS":
                        image = os.path.join(data_dir, file, file + "_t1.npy")
                        seg = os.path.join(data_dir, file, file + "_seg.npy")

                        if (image, seg) not in self.test_paths:
                            self.img_path.append([image, seg])
            textfile.close()
        else:
            self.img_path = test_paths
            if get_ventricles:
                for i , (path, seg) in enumerate(self.img_path):
                    file = path.split("/")[-1][:15]
                    map = os.path.join(root, "test_set_seg", file, file + "_seg.npy")
                    self.img_path[i] = [path, seg, map]
        self.return_name = return_name


    def __getitem__(self, index):

        if self.get_ventricles:
            img_path, seg_path, map_path = self.img_path[index]
            image = torch.from_numpy(np.load(img_path)[24:-24,24:-24, 5:-6]).unsqueeze(0).unsqueeze(0).float()/255.
            seg = torch.from_numpy(np.load(seg_path)[24:-24,24:-24, 5:-6]).unsqueeze(0).unsqueeze(0).float()
            map = torch.from_numpy(np.load(map_path)[24:-24, 24:-24, 5:-6]).unsqueeze(0).unsqueeze(0).float()
            seg[seg == 2.] = 1.
            seg[seg == 4.] = 1.
            if self.return_name:
                return torch.cat([image, seg, map]), img_path.split("/")[-1]
            else:
                return torch.cat([image, seg, map])
        else:
            img_path, seg_path = self.img_path[index]
            image = torch.from_numpy(np.load(img_path)[24:-24, 24:-24, 5:-6]).unsqueeze(0).unsqueeze(0).float() / 255.
            seg = torch.from_numpy(np.load(seg_path)[24:-24, 24:-24, 5:-6]).unsqueeze(0).unsqueeze(0).float()
            seg[seg == 2.] = 1.
            seg[seg == 4.] = 1.
            if self.return_name:
                return torch.cat([image, seg]), img_path.split("/")[-1]
            else:
                return torch.cat([image, seg])

    def __len__(self):
        return len(self.img_path)

def load_brats_2020(device, use_segmentation, test_size=40):
    target_img = torch.from_numpy(
        np.transpose(np.load("../brats_2020_2D/healthy/BraTS20_Training_019/BraTS20_Training_019_t1ce.npy"))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
    source_list = []
    for image in os.listdir('../brats_2020_2D/cancerous'):
        if image[:5] == "BraTS":
            if use_segmentation:
                source_seg = np.transpose(np.load("../brats_2020_2D/cancerous/" + image + "/" + image + "_seg.npy")).astype(float)
                source_seg[source_seg == 2.] = 1.
                source_seg[source_seg == 4.] = 1.
                source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                source_list.append(torch.cat([torch.from_numpy(
                    np.transpose(np.load('../brats_2020_2D/cancerous/' + image + "/" + image + "_t1ce.npy"))).type(
                    torch.FloatTensor).unsqueeze(0).unsqueeze(0), source_seg]))
            else:
                source_list.append(torch.from_numpy(
                    np.transpose(np.load('../brats_2020_2D/cancerous/' + image + "/" + image + "_t1ce.npy"))).type(
                    torch.FloatTensor).unsqueeze(0))

    random.shuffle(source_list)
    test_list = source_list[-test_size:]
    source_list = source_list[:-test_size]

    return torch.stack(source_list), torch.stack(test_list), target_img

