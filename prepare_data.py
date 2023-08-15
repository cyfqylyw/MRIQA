import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchio as tio


# nii -> npy, origin data
# nii2npy, result in origin data with npy format
file_lst = list(pd.read_csv('label.csv')['file_list'])

min_shape = (156, 156, 64)
transform = tio.Compose([tio.transforms.CropOrPad(min_shape)])

for file in file_lst:
    split_lst = file.split('/')
    ds_id, sub_id = split_lst[1], split_lst[2]
    if ds_id == 'ds000201_R1.0.5':
        sec_id = split_lst[3]
        data = nib.load(file).get_fdata()
        data = transform(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))
        np.save('./datasets/origin_data/' + ds_id + '__' + sub_id + '__' + sec_id + '.npy', data)
    else:
        data = nib.load(file).get_fdata()
        data = transform(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))
        np.save('./datasets/origin_data/' + ds_id + '__' + sub_id + '.npy', data)



# data argment: https://torchio.readthedocs.io/transforms/augmentation.html
# transform 1
transform_1 = tio.Compose([
        tio.RandomAffine(                # 随机仿射变换
            scales=(0.9, 1.1),           # 缩放，范围为0.9到1.1
            degrees=(-5, 5),             # 旋转，角度范围为-10到10度
            translation=(-5, 5),         # 平移，范围为-10到10像素
            isotropic=True,              # 如果为True，则3个轴使用相同的缩放值
        ),
        tio.RandomFlip(axes=(0, 1, 2)),  # 随机沿各轴翻转
    ])

ori_file_lst = os.listdir('./datasets/origin_data/')
for ori_file in ori_file_lst:
    ori_data = np.load('./datasets/origin_data/' + ori_file)
    data = transform_1(ori_data)
    np.save('./datasets/transformed_data_1/' + ori_file, data)


# transform 2
transform_2 = tio.Compose([
        tio.RandomMotion(degrees=2),
        tio.RandomGhosting(),
        # tio.RandomBiasField(),
        tio.RandomAnisotropy(axes=(0,1,2), downsampling=(1,2))
    ])

ori_file_lst = os.listdir('./datasets/origin_data/')
for ori_file in ori_file_lst:
    ori_data = np.load('./datasets/origin_data/' + ori_file)
    data = transform_2(ori_data)
    np.save('./datasets/transformed_data_2/' + ori_file, data)



# Fourlier transform

ori_file_lst = os.listdir('./datasets/origin_data/')
for ori_file in ori_file_lst:
    ori_data = np.load('./datasets/origin_data/' + ori_file)
    # apply Fourlier transform
    fourliered_data = np.fft.fftn(ori_data, axes=(0,1,2))
    real, imag = fourliered_data.real, fourliered_data.imag
    # three channels
    real_channel = np.sign(real) * np.log1p(np.abs(real))
    imag_channel = np.sign(imag) * np.log1p(np.abs(imag))
    abs_channel = np.log1p(np.abs(fourliered_data))
    # stack three channels
    fourlier_data = np.stack((real_channel, imag_channel, abs_channel), axis=0)
    np.save('./datasets/fourlier_origin_data/' + ori_file, fourlier_data)


file_lst = os.listdir('./datasets/transformed_data_1/')
for filename in file_lst:
    trans_data = np.load('./datasets/transformed_data_1/' + filename)
    # apply Fourlier transform
    fourliered_data = np.fft.fftn(trans_data, axes=(0,1,2))
    real, imag = fourliered_data.real, fourliered_data.imag
    # three channels
    real_channel = np.sign(real) * np.log1p(np.abs(real))
    imag_channel = np.sign(imag) * np.log1p(np.abs(imag))
    abs_channel = np.log1p(np.abs(fourliered_data))
    # stack three channels
    fourlier_data = np.stack((real_channel, imag_channel, abs_channel), axis=0)
    np.save('./datasets/fourlier_data_1/' + filename, fourlier_data)


file_lst = os.listdir('./datasets/transformed_data_2/')
for filename in file_lst:
    trans_data = np.load('./datasets/transformed_data_2/' + filename)
    # apply Fourlier transform
    fourliered_data = np.fft.fftn(trans_data, axes=(0,1,2))
    real, imag = fourliered_data.real, fourliered_data.imag
    # three channels
    real_channel = np.sign(real) * np.log1p(np.abs(real))
    imag_channel = np.sign(imag) * np.log1p(np.abs(imag))
    abs_channel = np.log1p(np.abs(fourliered_data))
    # stack three channels
    fourlier_data = np.stack((real_channel, imag_channel, abs_channel), axis=0)
    np.save('./datasets/fourlier_data_2/' + filename, fourlier_data)
