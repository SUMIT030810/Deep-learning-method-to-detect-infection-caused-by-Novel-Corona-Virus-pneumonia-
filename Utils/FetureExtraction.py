import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from Utils import LoadNSave


class FeatureExtractor():
    def __init__(self,device,model,pretrained=False):
        self.device=device
        self.model=model.to(device)
        self.pretrained=pretrained
        self.train_features = []
        self.train_labels = []
        self.val_features = []
        self.val_labels = []
        self.test_features = []
        self.test_labels = []
        self.load_n_save = LoadNSave()

    def extract_features(self,data_loader,saved_model_root_dir,model_name,model_file_name):
        if self.pretrained:
            model_state_dict=self.load_n_save.load_state(saved_model_root_dir,
                                                    model_name,
                                                    model_file_name,
                                                    self.pretrained)
            self.model.load_state_dict(model_state_dict['model'])
        print(f'Extracting features from saved {model_name}')
        features_list=[]
        labels_list=[]
        self.model.fc = nn.Identity()  # Removing the final layer to get feature embeddings
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)  # Get feature vectors
                features_list.append(outputs.cpu().numpy())
                labels_list.append(labels.numpy())
        # Convert lists to numpy arrays
        features_array = np.concatenate(features_list)
        labels_array = np.concatenate(labels_list)
        return features_array, labels_array

    def get_features(self,
                     data_loader,
                     saved_model_root_dir,
                     model_name,
                     model_file_name,
                     saved_feature_root_dir,
                     feature_file_name,
                     label_file_name):
        flag=self.load_n_save.check_if_features_extracted(saved_feature_root_dir,
                                                     feature_file_name,
                                                     label_file_name,
                                                     model_name)
        if not flag==1:
            features,labels=self.extract_features(data_loader,
                                                  saved_model_root_dir,
                                                  model_name,
                                                  model_file_name)
            self.load_n_save.save_npy(saved_feature_root_dir,
                                      model_name,
                                      feature_file_name,
                                      label_file_name,
                                      features,
                                      labels)
            return features,labels
        else:
            print(f'Loading features and labels from saved {feature_file_name} and {label_file_name}')
            features,labels=self.load_n_save.load_npy(saved_feature_root_dir,
                                                      feature_file_name,
                                                      label_file_name,
                                                      model_name)
            return features, labels