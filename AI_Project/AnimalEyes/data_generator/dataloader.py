import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
import unicodedata
from torch.utils.data import Dataset
from collections import Counter
from utils.data_preprocess import DataSpliter
import unicodedata

# root_dir = '/Users/minsu/Library/Mobile Documents/com~apple~CloudDocs/Animals/AnimalEyes/'
# csv_dir_path = os.path.join(root_dir , 'data')
# data_dir = '/Volumes/MINDB/AnimalEyes/'


class DogEyesDataset(Dataset):
    def __init__(self, transform = None, train = True):
        ## root setting
        self.root_dir = '/Users/minsu/Library/Mobile Documents/com~apple~CloudDocs/Animals/AnimalEyes/'
        self.csv_dir_path = os.path.join(self.root_dir , 'data')
        self.data_dir = '/Volumes/MINDB/AnimalEyes/'
        self.transform = transform

        self.disease_cls = [unicodedata.normalize('NFC', f) for f in os.listdir(os.path.join(self.data_dir, 'Training'))]

        data = DataSpliter(self.root_dir, self.csv_dir_path , self.data_dir)
        
        if train:
            self.image_df = data.train_set
            # print('train dataframe shape : ', self.image_df.shape)
        else:
            self.image_df = data.val_set
            # print('validation dataframe shape : ', self.image_df.shape)
                    
    def __len__(self):
        return len(self.image_df)
    
    def __getitem__(self, idx):
        image_path = self.image_df.iloc[idx, 0]

        image = Image.open(image_path)

        labels = torch.zeros(15)
        disease_name = self.image_df.iloc[idx, 1]

        cls_idx = self.disease_cls.index(disease_name)
        labels[cls_idx] = 1

        if self.transform:
            image = self.transform(image)
        
        ## image, labels (diseases , lv1, lv2, lv3)
        return image, labels


        