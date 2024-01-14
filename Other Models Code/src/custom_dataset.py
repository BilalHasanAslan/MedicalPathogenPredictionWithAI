import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Grayscale, RandomResizedCrop, Compose, RandAugment, Normalize
import csv
import re
import os
from neural_augment import neural_style
from PIL import Image
import pandas as pd
from scipy.stats import shapiro, bartlett
from scipy.stats import mannwhitneyu
from itertools import combinations
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

'''
This script encapsulates the Custom Dataset class structure and behaviours - which the main function is responsible for creating.
This class allows for the .csv files to be parsed, linking labels to images and thus subsequently populating the data loaders.
This class also contains the implementation of the data augmentation methods
The code in this class is inspired by: https://www.kaggle.com/code/boascent/multi-label-image-classification-pytorch-gpu
'''

class CustomDataset(Dataset):
    def __init__(self, image_dir, csv_file, data_aug=None):
        self.image_dir = image_dir
        self.data, self.classes = self.parse_csv(csv_file)
        self.num_classes = len(self.classes)
        self.data_aug = data_aug
        
        #Initialize the normalization parameters to the precomputed dataset mean and std values
        if 'Elbow' in image_dir:
            normalize = Normalize(mean=[0.1256, 0.1256, 0.1256],std=[0.2226, 0.2226, 0.2226])
        elif 'Neck' in image_dir:
           normalize = Normalize(mean=[0.5468, 0.5468, 0.5468],std=[0.2548, 0.2548, 0.2548])

        #Augment images depending on the data_aug input parameter value
        if self.data_aug == "RandomCrop":
            self.transform = Compose([
                RandomResizedCrop(224),
                Grayscale(num_output_channels=3),
                ToTensor()
            ])
        elif self.data_aug == "RandAug":
            self.transform = Compose([
                Grayscale(num_output_channels=3),
                RandAugment(3,25),
                ToTensor()
            ])
        #This case operates on both normal and neural augment images
        else:
            self.transform = Compose([
                Grayscale(num_output_channels=3),
                ToTensor()
            ]) 

    #This function reads the CSV file and parses the class labels and data entries
    def parse_csv(self, csv_file, encoding='utf-8'):
        
        with open(csv_file, 'r', encoding=encoding) as file:
            reader = csv.reader(file)
            labels_pre = next(reader)[2:] 
            labels = []
            for label in labels_pre:
                temp = label
                if temp[-1] == "#":
                    temp = temp[:-2]
                labels.append(temp)
            data = []
            for row in reader:
                image_id = row[1]
                class_labels = [int(label) for label in row[2:]]
                data.append((image_id, class_labels))
        return data, labels
       

    #This function returns the total number of data entries available in the dataset
    def __len__(self):
        return len(self.data)

    #This function retrives individual data entries (image and associated labels from the dataset)
    #Image transformations are applied in this step
    def __getitem__(self, index):
        image_id, class_labels = self.data[index]

        #Image loading, handling for the different extension types in the image directories
        try:
            image_path = os.path.join(self.image_dir, image_id  + '.jpg')
            image = Image.open(image_path)
        except:
                image_path = os.path.join(self.image_dir, image_id + '.jpeg')
                image = Image.open(image_path)

        #Image resize a tensore conversion
        image = image.resize((224, 224))
        image = self.transform(image)

        #Label tensor conversion
        label_tensor = torch.tensor(class_labels, dtype=torch.float32)
        return image, label_tensor


