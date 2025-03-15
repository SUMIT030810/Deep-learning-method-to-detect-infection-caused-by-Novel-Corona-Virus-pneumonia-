import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from Utils import LoadNSave


class FitModel():
    def __init__(self,device,model,model_name,start_epoch=0,num_epochs=200,learning_rate=0.001,pretrained=False):
        self.device = device
        self.model = model.to(self.device)
        self.model_name = model_name
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.criterion = nn.CrossEntropyLoss()
        self.pretrained = pretrained

    def train_single_epoch(self, train_loader, epoch):
        print(f'Epoch[{epoch + 1}/{self.num_epochs}] Training in progress...')
        self.model.train()
        running_loss = torch.zeros(1).to(self.device)
        correct = torch.zeros(1).to(self.device)
        total = 0
        num_classes = 3
        for _, (images, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}/{self.num_epochs}",total=len(train_loader)):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs_n = self.model(images)
            outputs = outputs_n.reshape(-1, num_classes)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss
            correct += torch.eq(torch.max(outputs.data, 1)[1], labels).sum()
            total += labels.size(0)
        epoch_loss = running_loss.item() / len(train_loader)
        epoch_acc = correct.item() / total
        return epoch_loss, epoch_acc

    def validate_single_epoch(self,val_loader,epoch):
        print(f'Epoch[{epoch+1}/{self.num_epochs}] Validation in progress...')
        self.model.eval()
        running_loss = torch.zeros(1).to(self.device)
        correct = torch.zeros(1).to(self.device)
        total = 0
        num_classes = 3
        with torch.no_grad():
            for _, (images, labels) in tqdm(enumerate(val_loader), desc=f"Epoch {epoch + 1}/{self.num_epochs}",total=len(val_loader)):
                images, labels=images.to(self.device), labels.to(self.device)
                outputs_n = self.model(images)
                outputs = outputs_n.reshape(-1, num_classes)
                loss = self.criterion(outputs, labels)
                running_loss += loss
                correct += torch.eq(torch.max(outputs.data, 1)[1], labels).sum()
                total += labels.size(0)
            epoch_loss = running_loss.item() / len(val_loader)
            epoch_acc = correct.item() / total
        return epoch_loss, epoch_acc

    def train_model(self,
                    train_loader,
                    val_loader,
                    test_loader,
                    saved_model_root_dir,
                    model_file_name,
                    saved_results_root_dir,
                    result_file_name):
        train_losses=[]
        train_accs=[]
        val_losses = []
        val_accs = []
        test_losses = []
        test_accs = []
        best_val_acc = 0.0
        load_n_save=LoadNSave()
        if self.pretrained:
            model_state_dict=load_n_save.load_state(saved_model_root_dir,
                                                    self.model_name,
                                                    model_file_name,
                                                    self.pretrained)
            self.model.load_state_dict(model_state_dict['model'])
            self.optimizer.load_state_dict(model_state_dict['optimizer'])
            result_state_dict = load_n_save.load_state(saved_results_root_dir,
                                                       self.model_name,
                                                       result_file_name,
                                                       self.pretrained)
            best_val_acc = result_state_dict['best_val_acc']
            self.start_epoch = result_state_dict['epoch']
            train_losses = result_state_dict['train_losses']
            train_accs = result_state_dict['train_accuracies']
            val_losses = result_state_dict['val_losses']
            val_accs = result_state_dict['val_accuracies']
            test_losses=result_state_dict['test_losses']
            test_accs = result_state_dict['test_accuracies']
        for epochs in range(self.start_epoch,self.num_epochs):
            train_loss,train_acc=self.train_single_epoch(train_loader,epochs)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_loss,val_acc=self.validate_single_epoch(val_loader,epochs)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f'Epoch[{epochs+1}/{self.num_epochs}] Summery')
            print("_________________________")
            print(f'Train Loss:{train_loss}')
            print(f'Train Accuracy:{train_acc}')
            print(f'Validation Loss:{val_loss}')
            print(f'Validation Accuracy:{val_acc}')
            if val_acc > best_val_acc or ((epochs+1)%10)==0:
                best_val_acc = val_acc
                test_loss, test_acc = self.validate_single_epoch(test_loader, epochs)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                print(f'Epoch[{epochs + 1}/{self.num_epochs}] Test Summery')
                print("_________________________")
                print(f'Test Loss:{test_loss}')
                print(f'Test Accuracy:{test_acc}')
                model_state_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                load_n_save.save_state(saved_model_root_dir,
                                       self.model_name,
                                       model_file_name,
                                       model_state_dict)
                result_state_dict = {
                    'best_val_acc': best_val_acc,
                    'epoch': epochs + 1,
                    'train_losses': train_losses,
                    'train_accuracies': train_accs,
                    'val_losses': val_losses,
                    'val_accuracies': val_accs,
                    'test_losses': test_losses,
                    'test_accuracies': test_accs
                }
                load_n_save.save_state(saved_results_root_dir,
                                       self.model_name,
                                       result_file_name,
                                       result_state_dict)
            self.scheduler.step(val_acc)
            print(f'Epoch {epochs + 1}, Loss: {val_loss}, Learning rate: {self.scheduler.get_last_lr()}')


