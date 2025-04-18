from torchvision import datasets
import models as models
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import dataset as dataset
from tqdm import tqdm 
import csv


class TrainModel():

    def __init__(self, is_gpu=True):
        self.model = None
        self.arr_dataset = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.is_gpu = is_gpu
        self.device = torch.device('cuda' if is_gpu else 'cpu')
        self.hist = []
        self.best_loss = float('inf')
        self.best_acc = -1 * float('inf')

    def reset_metrics(self):
        self.best_loss = float('inf')
        self.best_acc = -1 * float('inf')

    def set_model(self, id_model='alexnet', num_classes=10):
        self.model = models.get_model(name_model=id_model, num_classes=num_classes)

    def load_weights(self, weights):
        print('Loading weights:', weights)
        weights += '.pt'
        self.model.load_state_dict(torch.load(weights, map_location=self.device))

    def set_dataset(self, id_dataset='cifar100', batch_size=1024, num_train=45000, num_val=5000, download=False):
        print('Dataset Selected:',id_dataset,'| Batch:',batch_size,'|')
        self.arr_dataset = dataset.get_dataset(id_dataset=id_dataset, batch_size=batch_size, num_train=num_train, num_val=num_val, download=download)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def save_hist(self,path,fmt="%1.3f",delimiter=','):

        np.savetxt(path,np.asarray(self.hist),fmt=fmt,delimiter=delimiter,header='train_loss,train_acc,val_loss,val_acc')

    def evaluate_model(self, loader, generate_log=True, path_log='output', prefix=''):
        self.model.to(self.device)
        self.model.eval()
        arr_true = []
        arr_pred = []
        running_loss = 0
        test_acc = 0
        total = 0

        # Initialize progress bar
        progress_bar = tqdm(loader, desc='Evaluating')

        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            

            arr_pred.extend(predicted.cpu().detach().numpy())
            arr_true.extend(target.cpu().detach().numpy())
            total += target.size(0)
            test_acc += predicted.eq(target).sum().item()
                
            # Update progress bar description with current loss and accuracy
            progress_bar.set_description(f'Loss: {running_loss / total:.4f}, Acc: {test_acc / total:.4f}')

        # Calculate final metrics
        final_loss = running_loss / len(loader)
        final_acc = test_acc / total
        
        # Export results to CSV if logging is enabled
        if generate_log:
            log_file_path = f'{path_log}/{prefix}evaluation_log.csv'
            with open(log_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['True Label', 'Predicted Label'])
                for true, pred in zip(arr_true, arr_pred):
                    writer.writerow([true, pred])
                    
                # Optionally, you could add another section for summary statistics
                writer.writerow([])
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Loss', final_loss])
                writer.writerow(['Accuracy', final_acc])

        return final_loss, final_acc



    def train_model(self, epochs=100, weight_path='backlog', sufix='', patience=5):
        self.model.to(self.device)
        self.model.train()

        print('Training model', self.model.__class__.__name__)

        trainloader, testloader, _ = self.arr_dataset

        # Initialize counters for early stopping
        epochs_since_improvement_loss = 0
        epochs_since_improvement_acc = 0

        for epoch in range(epochs):
            running_loss = 0.0
            train_acc = 0.0
            total = 0.0

            progress_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                train_acc += predicted.eq(target).sum().item()

                progress_bar.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/total:.4f}, Acc: {train_acc/total:.4f}")

            val_loss, val_acc = self.evaluate_model(testloader, generate_log=False)
            self.hist.append([running_loss/len(trainloader), train_acc/total, val_loss, val_acc])

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Early stopping logic for validation loss
            if val_loss < self.best_loss:
                torch.save(self.model.state_dict(), f"{weight_path}/loss_weights_{sufix}.pt")
                torch.save(self.model, f"{weight_path}/loss_full_model_{sufix}.pt")
                self.best_loss = val_loss
                epochs_since_improvement_loss = 0  # Reset counter
            else:
                epochs_since_improvement_loss += 1

            # Early stopping logic for validation accuracy
            if val_acc > self.best_acc:
                torch.save(self.model.state_dict(), f"{weight_path}/acc_weights_{sufix}.pt")
                torch.save(self.model, f"{weight_path}/acc_full_model_{sufix}.pt")
                self.best_acc = val_acc
                epochs_since_improvement_acc = 0  # Reset counter
            else:
                epochs_since_improvement_acc += 1

            # Check if training should be stopped early
            if epochs_since_improvement_loss >= patience or epochs_since_improvement_acc >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break



