import nibabel as nib
from skimage.exposure import match_histograms
import numpy as np
#import matplotlib.pyplot as plt
#import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import os

def rigid_registration(input):
    MNI_nib = nib.load("/Users/maillard/Downloads/sri24_spm8/templates/T1_brain.nii")
    MNI_img = MNI_nib.get_fdata()

    source = nib.load("/Users/maillard/Downloads/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/BraTS2021_00008/BraTS2021_00008_T1.nii.gz")
    source_img = source.get_fdata()

    """source_nib = nib.load("/home/matthis/datasets/BraTS_2021/" + input + "/" + input + "_t1.nii.gz")
    source_img = source_nib.get_fdata()
    source_seg = nib.load("/home/matthis/datasets/BraTS_2021/" + input + "/" + input + "_seg.nii.gz").get_fdata()
    source_seg[source_seg > 0] = 1."""


    """MNI_img = np.pad(MNI_img, ((22,21),(4,3),(0,0)))
    source_img = np.pad(source_img, ((0,0),(0,0), (17,17)))
    source_seg = np.pad(source_seg, ((0,0),(0,0), (17,17)))"""


    source_img = match_histograms(source_img, MNI_img)
    source_img = (source_img - source_img.min()) / (source_img.max() - source_img.min())
    MNI_img = (MNI_img - MNI_img.min()) / (MNI_img.max() - MNI_img.min())
    #source_img = source_img[:, ::-1, :]
    #source_seg = source_seg[:, ::-1, :]

    #MNI = nib.Nifti1Image(MNI_img, MNI_nib.affine)
    #nib.save(MNI, "/Users/maillard/Downloads/sri24_spm8/templates/T1_brain_scaled.nii")

    source = nib.Nifti1Image(source_img, MNI_nib.affine)
    nib.save(source, "/Users/maillard/Downloads/sri24_spm8/templates/brats_image_test.nii")
    """source_s = nib.Nifti1Image(source_seg, MNI_nib.affine)
    nib.save(source_s, "/home/matthis/datasets/source_seg_45.nii.gz")

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.ReadImage("/home/matthis/datasets/nonLinMNI.nii.gz"))
    elastixImageFilter.SetMovingImage(sitk.ReadImage("/home/matthis/datasets/source_45.nii.gz"))
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage(), "/home/matthis/datasets/source_aligned_45.nii.gz")

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(sitk.ReadImage("/home/matthis/datasets/source_seg_45.nii.gz"))
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.LogToConsoleOn()
    transformixImageFilter.Execute()
    sitk.WriteImage(transformixImageFilter.GetResultImage(), "/home/matthis/datasets/source_seg_aligned_45.nii.gz")


    im = nib.load("/home/matthis/datasets/source_seg_aligned_45.nii.gz")
    data = im.get_fdata()
    data = gaussian_filter(data, 3)
    data[data>0.5] = 1
    im2 = nib.Nifti1Image(data[40:-40,20:-20,:-25], im.affine)


    im = nib.load("/home/matthis/datasets/source_aligned_45.nii.gz")
    mni = nib.load("/home/matthis/datasets/nonLinMNI.nii.gz")
    data1 = im.get_fdata()[40:-40,20:-20,:-25]
    data2 = mni.get_fdata()[40:-40,20:-20,:-25]

    data1[data1<1e-5] = 0.
    #data1[data1 > 0] = match_histograms(data1[data1 > 0], data2[data2 > 0])

    os.mkdir("/home/matthis/datasets/brats_preproc/" + input)
    nib.save(im2, "/home/matthis/datasets/brats_preproc/" + input + "/" + input + "seg.nii.gz")
    nib.save(nib.Nifti1Image(data1, mni.affine), "/home/matthis/datasets/brats_preproc/" + input + "/" + input + "t1.nii.gz")
    nib.save(nib.Nifti1Image(data2, mni.affine), "/home/matthis/datasets/target_final.nii.gz")"""

"""for image in os.listdir("/home/matthis/datasets/BraTS_2021"):
    rigid_registration(image)"""


def extract_slices():
    MNI_nib = nib.load("/home/matthis/Nextcloud/templates/T1_brain.nii")
    MNI_img = MNI_nib.get_fdata()
    data_dir = "/home/matthis/datasets/BraTS_2021/"
    if not os.path.exists("/home/matthis/datasets/BraTS2021_preproc"):
        os.mkdir("/home/matthis/datasets/BraTS2021_preproc")

    for file in os.listdir(data_dir):
        source_img = nib.load(data_dir + file + "/" + file + "_t1.nii.gz").get_fdata()

        source_img[source_img > 0] = match_histograms(source_img[source_img > 0], MNI_img[MNI_img > 0])
        source_img = (source_img - source_img.min()) / (source_img.max() - source_img.min())
        source_seg = nib.load(data_dir + file + "/" + file + "_seg.nii.gz").get_fdata()
        if not os.path.exists("/home/matthis/datasets/BraTS2021_preproc/" + file):
            os.mkdir("/home/matthis/datasets/BraTS2021_preproc/" + file)
        np.save("/home/matthis/datasets/BraTS2021_preproc/" + file + "/" + file + "_seg.npy", (source_seg).astype(np.uint8))
        np.save("/home/matthis/datasets/BraTS2021_preproc/" + file + "/" + file + "_t1.npy", (source_img*255).astype(np.uint8))

extract_slices()
