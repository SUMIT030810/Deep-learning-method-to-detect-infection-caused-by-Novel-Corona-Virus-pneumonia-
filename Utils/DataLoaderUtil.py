import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class DatasetLoader():
    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.num_test_samples=None
        self.num_test_batches=None
        self.num_train_samples =None
        self.num_train_batches = None
        self.num_val_samples = None
        self.num_val_batches = None

    def get_data_loader(self,transform=None,shuffle=False,sampler=None,batch_size=64):
        dataset = ImageFolder(root=self.root_dir, transform=transform)
        data_loader=DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,sampler=sampler)
        self.num_test_samples=len(dataset)
        self.num_test_batches=len(data_loader)
        return data_loader

    def get_train_val_loader(self,transform,ratio=0.8,shuffle=False,sampler=None,batch_size=64):
        dataset = ImageFolder(root=self.root_dir, transform=transform)
        train_size = int(ratio* len(dataset))
        val_size = len(dataset)-train_size
        train_dataset, val_dataset= torch.utils.data.random_split(dataset,[train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
        self.num_train_samples = len(train_dataset)
        self.num_train_batches = len(train_loader)
        self.num_val_samples = len(val_dataset)
        self.num_val_batches = len(val_loader)
        return train_loader,val_loader

    def get_dataset_report(self):
        if not self.num_train_samples is None and not self.num_val_samples is None:
            print(f'Number of training samples:{self.num_train_samples}')
            print(f'Number of training batches:{self.num_train_batches}')
            print(f'Number of validation samples:{self.num_val_samples}')
            print(f'Number of validation batches:{self.num_val_batches}')
        if not self.num_test_samples is None:
            print(f'Number of test samples:{self.num_test_samples}')
            print(f'Number of test batches:{self.num_test_batches}')