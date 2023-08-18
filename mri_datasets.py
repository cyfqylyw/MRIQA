import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader


def get_filename(file_path):
    """
    change file path: datasets/ds000030_R1.0.5/sub-10680/anat/sub-10680_T1w.nii.gz
    into file name: ds000030_R1.0.5__sub-10680.npy

    change file path: datasets/ds000201_R1.0.5/sub-9036/ses-1/anat/sub-9036_ses-1_T1w.nii.gz
    into file name: ds000201_R1.0.5__sub-9036__ses-1.npy
    """
    # split the file path
    split_lst = file_path.split('/')
    ds_id, sub_id = split_lst[1], split_lst[2]
    sec_id = split_lst[3] if (ds_id == 'ds000201_R1.0.5') else None
    # obtain filename through string concatenation 
    filename = (ds_id + '__' + sub_id + '.npy') if (sec_id is None) else (ds_id + '__' + sub_id + '__' + sec_id + '.npy')
    return filename


class MRI_Dataset(Dataset):
    """
    get Dataset of the origin version and Fourlier version
    : __getitem__ : will return two torch.Tensor() with shape (1, 156, 156, 64)
    # i.e. X, Fourlier(X)
    # one is the origin MRI
    # another is after Fourlier transformation
    """
    def __init__(self, file_lst):
        self.file_lst = file_lst

    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, idx):
        filename = get_filename(file_path=self.file_lst[idx])
        origin_data = np.load('datasets/origin_data/' + filename)
        fourlier_data = np.load('datasets/fourlier_origin_data/' + filename)
        return torch.tensor(origin_data), torch.tensor(fourlier_data)
    

class MRI_Dataset_augmentation(Dataset):
    """
    get Dataset after augmentation transformation
    : __getitem__ : will return two torch.Tensor() with shape (1, 156, 156, 64)
    # i.e. aug1(X), aug2(X)
    # one is after RandomAffine, RandomFlip
    # another is after RandomMotion, RandomGhosting, RandomAnisotropy
    """
    def __init__(self, file_lst):
        self.file_lst = file_lst

    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, idx):
        filename = get_filename(file_path=self.file_lst[idx])
        transformed_data_1 = np.load('datasets/transformed_data_1/' + filename)
        transformed_data_2 = np.load('datasets/transformed_data_2/' + filename)
        return torch.tensor(transformed_data_1), torch.tensor(transformed_data_2)


class MRI_Dataset_Fourlier(Dataset):
    """
    get Dataset after augmentation transformation followed by Fourlier transformation
    : __getitem__ : will return two torch.Tensor() with shape (1, 156, 156, 64)
    # i.e. Fourlier(aug1(X)), Fourlier(aug2(X))
    # each are after two different augmentation transformation
    # both are after Fourlier transformation
    """
    def __init__(self, file_lst):
        self.file_lst = file_lst

    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, idx):
        filename = get_filename(file_path=self.file_lst[idx])
        fourlier_data_1 = np.load('datasets/fourlier_data_1/' + filename).reshape((3, 156,156,64)).astype(np.float32)
        fourlier_data_2 = np.load('datasets/fourlier_data_2/' + filename).reshape((3, 156,156,64)).astype(np.float32)
        return torch.tensor(fourlier_data_1), torch.tensor(fourlier_data_2)


def get_dataloader(mode, batch_size, seed=42, train=True, train_prop=0.9, shuffle=False):
    """
    return DataLoader based on the input parameters 
    :param mode: one choise from ['origin', 'augmentation', 'fourlier'], apply the three Dataset above respectively
    :param train: if train=True, use 90% of the data; else, use the rest 10% of the data
    :param train_prop: the proportion used for training
    :param batch_size: batch size
    :param shuffle: whether shuffle or not
    """
    random.seed(seed)
    torch.manual_seed(seed)

    file_lst = list(pd.read_csv('label.csv')['file_list'])
    file_lst_train = random.sample(file_lst, int(train_prop * len(file_lst)))
    file_lst_val = [file for file in file_lst if file not in file_lst_train]
    file_lst_input = file_lst_train if train else file_lst_val

    if mode == 'origin':
        dataset = MRI_Dataset(file_lst=file_lst_input)
    elif mode == 'augmentation':
        dataset = MRI_Dataset_augmentation(file_lst=file_lst_input)
    elif mode == 'fourlier':
        dataset = MRI_Dataset_Fourlier(file_lst=file_lst_input)
    else:
        raise TypeError("mode is not valid")

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

    return dataloader
