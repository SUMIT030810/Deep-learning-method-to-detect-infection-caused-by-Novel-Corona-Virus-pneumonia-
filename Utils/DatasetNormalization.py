import os
from os import mkdir

import torch
from torchvision import transforms
from tqdm import tqdm

from Utils import DatasetLoader, LoadNSave


class MeanStd():
    def __init__(self,device):
        self.mean = 0.0
        self.std = 0.0
        self.device = device
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(), ])

    def calculate_mean_std(self,root_dir,batch_size=64):
        dataset_loader=DatasetLoader(root_dir)
        data_loader,_=dataset_loader.get_data_loader(transform=self.transform,batch_size=batch_size)
        sum_rgb = torch.zeros(3)
        sum_sq_rgb = torch.zeros(3)
        num_pixels = 0
        print(f"Calculating Mean and Standard Deviation on the Dataset...")
        for images, _ in tqdm(data_loader):
            images.to(self.device)
            num_batch_pixels = images.size(0) * images.size(2) * images.size(3)
            sum_rgb += images.sum([0, 2, 3])
            sum_sq_rgb += (images ** 2).sum([0, 2, 3])
            num_pixels += num_batch_pixels
        mean = sum_rgb / num_pixels
        std = torch.sqrt((sum_sq_rgb / num_pixels) - (mean ** 2))
        print(f"Mean for each channel: {mean}")
        print(f"Standard deviation for each channel: {std}")
        return mean, std

    def calculate_mean_std_v1(self,root_dir,batch_size=64):
        dataset_loader=DatasetLoader(root_dir)
        data_loader=dataset_loader.get_data_loader(transform=self.transform,batch_size=batch_size)
        mean = torch.zeros(3)  # RGB channels
        std = torch.zeros(3)

        print(f"Calculating Mean and Standard Deviation on the Dataset...")
        for images, _ in tqdm(data_loader):
            images.to(self.device)
            # Flatten the images across all batches
            # images shape: (batch_size, 3, H, W)
            batch_mean = images.mean(dim=[0, 2, 3])  # Mean per channel
            batch_std = images.std(dim=[0, 2, 3])  # Std per channel
            mean += batch_mean
            std += batch_std
        mean /= len(data_loader)
        std /= len(data_loader)
        print(f"Mean for each channel: {mean}")
        print(f"Standard deviation for each channel: {std}")
        return mean, std

    def get_mean_std(self,flag,dataset_root_dir,saved_root_dir,file_name,batch_size=64):
        load_n_save=LoadNSave()
        if flag==0:
            os.mkdir(saved_root_dir)
            self.mean,self.std=self.calculate_mean_std_v1(dataset_root_dir,batch_size)
            load_n_save.save_mean_std(saved_root_dir,file_name,self.mean,self.std)
        else:
            if flag==-1:
                self.mean,self.std = self.calculate_mean_std_v1(dataset_root_dir, batch_size)
                load_n_save.save_mean_std(saved_root_dir, file_name, self.mean, self.std)
            else:
                self.mean,self.std=load_n_save.load_mean_std(saved_root_dir,file_name)
                print(f"Mean for each channel: {self.mean}")
                print(f"Standard deviation for each channel: {self.std}")
        return self.mean,self.std
