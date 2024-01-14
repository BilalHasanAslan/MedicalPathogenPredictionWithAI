import torch
import torchvision.models as tv_models
import torch.nn as nn
import torch.optim as optim
from custom_dataset import CustomDataset as cd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import time
import numpy as np
from sklearn.metrics import hamming_loss, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import argparse
from PIL import Image

'''
Author: Winner Kazaka
Orginisation: UCT
This code encapsulates data loading, model creation and training pipeline implementations delineated in the paper.
This code is inspired by https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load datasets and create dataLoaders
def load_data(dir_path, batch_size=32, shuffle_train=True, data_aug=None):
    train_dataset = cd(f"{dir_path}/images", f"{dir_path}/train.csv", data_aug=data_aug)
    val_dataset = cd(f"{dir_path}/images", f"{dir_path}/val.csv")

    image_datasets = {
        'train': train_dataset,
        'val': val_dataset
    }

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train),
        'val': DataLoader(val_dataset, batch_size=batch_size)
    }
    
    return dataloaders, image_datasets

# Create and return model
def create_model(model_name, num_classes, tl_method):
    print(num_classes)
    #If No TL set all parameters to tunable
    if tl_method == "not_pretrained":
        model = tv_models.get_model(model_name, weights=None).to(device)
        for param in model.parameters():
            param.requires_grad = True 
    #Otherwise initialize the weights to the Image1K implementation and freeze the weights
    else:
        print(model_name)
        model = tv_models.get_model(model_name, weights="DEFAULT").to(device)
        for param in model.parameters():
            param.requires_grad = False 

    #Replace the classification layers - Handles exceptions for the different architecture types
    try:
       num_ftrs = model.classifier[-1].in_features
       model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        ).to(device)
    except:
        try:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_ftrs, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            ).to(device)
        except:
            try:
                num_ftrs = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(num_ftrs, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, num_classes)
                ).to(device)
            except:
                num_ftrs = model.head.in_features
                model.head = nn.Sequential(
                    nn.Linear(num_ftrs, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, num_classes)
                ).to(device)       
    return model

#Train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, image_datasets, num_classes, model_name, dir_path, da_method, iteration=None):
    since = time.time()
    save_name = model_name + "_" + dir_path[-4:] + ".pt"
    best_model_params_path = os.path.join("../trained_models/", save_name)
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0
    best_ovr_acc = 0.0
    best_hamming = float('inf') 
    best_f1 = 0.0  
    best_auc = 0.0 
    best_precision = 0.0 
    best_recall = 0.0  

    #metrics = {'train': {'loss': [], 'acc': [], 'hamming': [], 'f1': [], 'auc': [], 'precision': [], 'recall': [], 'per_label_acc': [], 'overall_acc': []},
    #           'val': {'loss': [], 'acc': [], 'hamming': [], 'f1': [], 'auc': [], 'precision': [], 'recall': [], 'per_label_acc': [], 'overall_acc': []}}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase, dataloader in dataloaders.items():
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_predictions = 0

            all_labels = []
            all_preds = []

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                #The Exact Match implementation is inspired by https://gist.github.com/jadhavpritish/539167245e7f4ff7d5a0095b7d7be2db#file-emr_evaluation_metric-py
                #Implements a threshold of 0.5
                probabilities = torch.sigmoid(outputs)
                preds = (probabilities > 0.5).float()
               
                #Correct predictions calculated using Exact Match (all labels predicted correctly)
                batch_corrects = torch.sum(preds == labels.data, dim=1) == num_classes
                total_predictions += len(preds)

                running_corrects += batch_corrects.sum().item()
                running_loss += loss.item() * inputs.size(0)

                all_labels.append(labels.cpu().detach().numpy())
                all_preds.append(preds.cpu().detach().numpy())

            if phase == 'train':
                scheduler.step() 
            
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)

            epoch_loss = running_loss / total_predictions
            epoch_acc = running_corrects / total_predictions

            print(save_name)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            f1 = f1_score(all_labels, all_preds, average='macro')
            precision = precision_score(all_labels, all_preds, average='micro')
            recall = recall_score(all_labels, all_preds, average='micro')

            per_label_acc = np.mean(all_labels == all_preds, axis=0)
            overall_acc = np.mean(per_label_acc)

            #Early stoppping implementation - save the model at the epoch with the highest validation accuracy
            if phase == 'val':
                best_f1 = max(best_f1, f1)
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), best_model_params_path)

    time_elapsed = time.time() - since

    print(save_name)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    print(f'Best ovr per label Acc: {best_ovr_acc:4f}')
    print(f'Best Hamming Loss: {best_hamming:.4f}')
    print(f'Best F1: {best_f1:.4f}')
    print(f'Best AUC: {best_auc:.4f}')
    print(f'Best Precision: {best_precision:.4f}')
    print(f'Best Recall: {best_recall:.4f}')
    model.load_state_dict(torch.load(best_model_params_path))   

#Main execution
def main(config):
    criterion = nn.BCEWithLogitsLoss()
    model_names = config.model_names.split(',')  
    dir_paths = config.dir_paths.split(',')  
    data_augmentations = config.data_augmentations.split(',')  
    transfer_learning_methods = config.transfer_learning_methods.split(',')
    #Populate parameters for model creation
    for da_method in data_augmentations:
        for dir_path in dir_paths:
            dataloaders, image_datasets = load_data(dir_path, data_aug=da_method) 
            num_classes = len(image_datasets['train'].classes)
            for tl_method in transfer_learning_methods:
                num_epochs = 15 if tl_method == "not_pretrained" else 10
                for model_name in model_names:
                    model = create_model(model_name, num_classes, tl_method) #Create model with distict parameters
                    #Initialize hyperparameters
                    out_name = f"{model_name}_{tl_method}_{da_method}"
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0001)
                    train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, image_datasets, num_classes, out_name, dir_path, da_method)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', type=str, default='convnext_base')
    parser.add_argument('--dir_paths', type=str, default='./dataset/Neck')
    parser.add_argument('--data_augmentations', type=str, default='RandomCrop')
    parser.add_argument('--transfer_learning_methods', type=str, default='pretrained')
    config = parser.parse_args()
    main(config)