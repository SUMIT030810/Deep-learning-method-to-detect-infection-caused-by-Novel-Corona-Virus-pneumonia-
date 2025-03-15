import json
import torch

from Models import resnet50
from Models.MobileNet import MobileNet, MobileNet_Model
from Models.AlexNet import  AlexNet_Model
# from Models.AlexNet_TransferLearning import AlexNet_TransferLearning
from Models.VGG_16 import VGG_16
from Models.EfficientNet import EfficientNet, EfficientNetModel
from Utils import get_device, display_device_info, MeanStd, LoadNSave, get_train_transform, get_test_transform,DatasetLoader, FitModel

#Select available device
device=get_device()
display_device_info(device)

#Loading configuration
with open('project_config.json','r') as file:
    config=json.load(file)
num_classes=config['num_classes']
train_root_dir=config['train_root_dir']
test_root_dir=config['test_root_dir']
batch_size=config['batch_size']
saved_mean_std_root_dir=config['mean_std_root_dir']
train_mean_std_filename=config['saved_train_mean_std_filename']
test_mean_std_filename=config['saved_test_mean_std_filename']
saved_model_root_dir=config['saved_model_root_dir']
saved_results_root_dir=config['saved_results_root_dir']
saved_model_filename=config['saved_model_filename']
saved_results_filename=config['saved_results_filename']
num_epochs=config['num_epochs']
learning_rate=config['learning_rate']
saved_features_root_dir=config['saved_features_root_dir']
train_features_filename=config['train_features_filename']
val_features_filename=config['val_features_filename']
test_features_filename=config['test_features_filename']
train_labels_filename=config['train_labels_filename']
val_labels_filename=config['val_labels_filename']
test_labels_filename=config['test_labels_filename']

#Creating model
model,model_name=AlexNet_Model(num_classes)
pretrained=False
#Test model
# x=torch.randn(1,3,224,224)
x=torch.randn(1,3,227,227)
y=model(x)
print(f'Model:{model_name}')
print(f'Shape:{y.shape}')

#Dataset normalization
load_n_save=LoadNSave()
dataset_mean_std=MeanStd(device)
flag=load_n_save.check_if_exist(saved_mean_std_root_dir,train_mean_std_filename)
train_dataset_mean,train_dataset_std=dataset_mean_std.get_mean_std(flag,train_root_dir,saved_mean_std_root_dir,train_mean_std_filename,batch_size)
flag=load_n_save.check_if_exist(saved_mean_std_root_dir,test_mean_std_filename)
test_dataset_mean,test_dataset_std=dataset_mean_std.get_mean_std(flag,test_root_dir,saved_mean_std_root_dir,test_mean_std_filename,batch_size)

#Dataset preprocessing
train_transform=get_train_transform(train_dataset_mean,train_dataset_std)
test_transform=get_test_transform(test_dataset_mean,test_dataset_std)

#Loading Dataset
train_dataset_loader=DatasetLoader(train_root_dir)
train_loader,val_loader=train_dataset_loader.get_train_val_loader(train_transform,batch_size=batch_size)
train_dataset_loader.get_dataset_report()
test_dataset_loader=DatasetLoader(test_root_dir)
test_loader=test_dataset_loader.get_data_loader(transform=test_transform,batch_size=batch_size)
test_dataset_loader.get_dataset_report()

#Train model
model_name_string=f'{saved_model_filename}_{model_name}'
flag_model=load_n_save.check_if_exist(saved_model_root_dir,model_name_string)
result_name_string=f'{saved_results_filename}_{model_name}'
flag_result=load_n_save.check_if_exist(saved_results_root_dir,result_name_string)
#print(f'model:{flag_model} result:{flag_result}')
if flag_model==1 and flag_result==1:
    pretrained=True
fit_model=FitModel(device,
                   model,
                   model_name,
                   num_epochs=num_epochs,
                   learning_rate=learning_rate,
                   pretrained=pretrained)

fit_model.train_model(train_loader,
                      val_loader,
                      test_loader,
                      saved_model_root_dir,
                      saved_model_filename,
                      saved_results_root_dir,
                      saved_results_filename)

