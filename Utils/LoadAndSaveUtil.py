import os

import numpy as np
import torch


class LoadNSave():
    def __init__(self):
        self.ext='.pth'

    def check_if_exist(self,root_dir,file_name):
        file_path = f'{root_dir}/{file_name}{self.ext}'
        if not os.path.isdir(root_dir):
            print(f'No directory named {root_dir} found!!!')
            return 0
        else:
            if not os.path.isfile(file_path):
                print(f'File {file_path} not found!!!')
                return -1
            else:
                return 1
    def check_if_features_extracted(self,root_dir,feature_file_name,label_file_name,model_name):
        feature_file_path = f'{root_dir}/{feature_file_name}_{model_name}.npy'
        label_file_path = f'{root_dir}/{label_file_name}_{model_name}.npy'
        if not os.path.isdir(root_dir):
            print(f'No directory named {root_dir} found!!!')
            return 0
        else:
            if not os.path.isfile(feature_file_path) or not os.path.isfile(label_file_path):
                print(f'.npy file not found!!!')
                return -1
            else:
                return 1

    def save_mean_std(self,root_dir,file_name,mean,std):
        file_path = f'{root_dir}/{file_name}{self.ext}'
        try:
            torch.save({'mean': mean, 'std': std}, file_path)
            print(f"Mean and std results saved successfully to {file_path}")
        except IOError as ioe:
            print(f"Failed to save Results due to an IOError: {ioe}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the result: {e}")

    def load_mean_std(self,root_dir,file_name):
        file_path = f'{root_dir}/{file_name}{self.ext}'
        mean_std = None
        print(f"Loading dataset mean standard...")
        try:
            mean_std = torch.load(file_path, weights_only=True)
        except Exception as e:
            print(f"ERROR(Fail to load parameters):{e}")
        mean = mean_std['mean']
        std = mean_std['std']
        return mean, std

    def save_state(self,root_dir,model_name,file_name,state):
        name=f'{file_name}_{model_name}'
        flag=self.check_if_exist(root_dir,name)
        if flag==0:
            os.mkdir(root_dir)
        print(f"Saving model parameters...")
        file_path = f'{root_dir}/{file_name}_{model_name}{self.ext}'
        try:
            torch.save(state, file_path)
            print(f"Model saved successfully to {file_path}")
        except IOError as ioe:
            print(f"Failed to save model due to an IOError: {ioe}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the model: {e}")

    def load_state(self,root_dir,model_name,file_name,pretrained):
        state_dict = {}
        file_path = f'{root_dir}/{file_name}_{model_name}{self.ext}'
        if pretrained:
            print(f"Loading from saved model...")
            assert os.path.isdir(root_dir), 'Error: no checkpoint directory found!'
            try:
                state_dict = torch.load(file_path, weights_only=True)
            except Exception as e:
                print(f"ERROR(Fail to load parameters):{e}")
        return state_dict

    def save_npy(self,root_dir,model_name,feature_file_name,label_file_name,feature_vector,label_vector):
        if not os.path.isdir(root_dir):
            print(f'No directory named {root_dir} found!!!')
            os.mkdir(root_dir)
        feature_file_path = f'{root_dir}/{feature_file_name}_{model_name}'
        label_file_path = f'{root_dir}/{label_file_name}_{model_name}'
        np.save(feature_file_path, feature_vector)
        np.save(label_file_path, label_vector)
        print(f'Features and labels saved as {feature_file_path} and {label_file_path}')

    def load_npy(self,root_dir,feature_file_name,label_file_name,model_name):
        feature_file_path = f'{root_dir}/{feature_file_name}_{model_name}.npy'
        label_file_path = f'{root_dir}/{label_file_name}_{model_name}.npy'
        print(f"Loading features and labels from saved array...")
        assert os.path.isdir(root_dir), 'Error: no checkpoint directory found!'
        try:
            feature_vector = np.load(feature_file_path)
            label_vector = np.load(label_file_path)
        except Exception as e:
            print(f"ERROR(Fail to load features):{e}")
        return feature_vector, label_vector

