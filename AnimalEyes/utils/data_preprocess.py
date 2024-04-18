import torch
import os
import cv2
import json
import pandas as pd
import unicodedata
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

# root_dir = '/Users/minsu/Library/Mobile Documents/com~apple~CloudDocs/Animals/AnimalEyes/'
# csv_dir_path = os.path.join(root_dir , 'data')
# data_dir = '/Volumes/MINDB/AnimalEyes/'


class DataSpliter():
    def __init__(self , root_dir, csv_dir_path , data_dir):
        ## root setting
        self.root_dir = root_dir
        self.csv_dir_path = csv_dir_path
        self.data_dir = data_dir

        self.make_csv_dir(self.csv_dir_path)
        
        if not os.path.exists(os.path.join(self.csv_dir_path, 'image_df.csv')):
            self.image_df = self.get_image_df(self.data_dir)
            self.image_df.to_csv(os.path.join(self.csv_dir_path, 'image_df.csv'), encoding='utf-8', index = False)

            splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

            for train_idx, val_idx in splitter.split(self.image_df , self.image_df['name']):
                self.train_set = self.image_df.iloc[train_idx]
                self.val_set = self.image_df.iloc[val_idx]

            self.train_set.to_csv(os.path.join(self.csv_dir_path, 'train_image_df.csv'), encoding='utf-8', index = False)
            self.val_set.to_csv(os.path.join(self.csv_dir_path, 'val_image_df.csv'), encoding='utf-8', index = False)
        else:
            self.image_df = pd.read_csv(os.path.join(self.csv_dir_path, 'image_df.csv'), encoding='utf-8')
            self.train_set = pd.read_csv(os.path.join(self.csv_dir_path, 'train_image_df.csv'), encoding='utf-8')
            self.val_set = pd.read_csv(os.path.join(self.csv_dir_path, 'val_image_df.csv'), encoding='utf-8')

    def make_csv_dir(self, csv_dir_path):
        if not os.path.exists(csv_dir_path):
            os.mkdir(csv_dir_path)

    def get_image_df(self, train = True):
        extensions = ['.json']
        image_df = pd.DataFrame()
        orders = {'image_paths' : 1 , 'name' : 2, 'lv1' : 3, 'lv2' : 4, 'lv3' : 5}

        if train:
            data_dir = os.path.join(self.data_dir, 'Training')
        else:
            data_dir = os.path.join(self.data_dir, 'Test')

        for root, dirs , files in tqdm(os.walk(data_dir)):
            for file in files:
                if file.lower().endswith(extensions[0]):
                    with open(os.path.join(root,file) , 'r', encoding = 'UTF-8') as f:
                        image_info = json.load(f)

                    labels = self.get_labels(image_info)
                    labels['image_paths'] = os.path.join(root, 'crop_' + image_info['images']['meta']['file_name'])

                    labels = {k: labels[k] for k in sorted(labels, key=orders.get)}
                    image_df = pd.concat([image_df, pd.DataFrame(labels, index = ['0'])], axis = 0 , ignore_index = True)
                    
        return image_df

    def get_labels(self, image_info):
        labels = {}
        labels['lv1'] = image_info['label']['label_disease_lv_1']
        labels['lv2'] = image_info['label']['label_disease_lv_2']
        labels['lv3'] = image_info['label']['label_disease_lv_3']
        counter = Counter([labels['lv1'], labels['lv2'], labels['lv3']])

        if counter.most_common(1)[0][0] == '무':
            labels['name'] = '무'
        elif counter.most_common(1)[0][0] == '유':
            labels['name'] = image_info['label']['label_disease_nm']
        else:
            labels['name'] = image_info['label']['label_disease_nm'] + '_' + counter.most_common(1)[0][0]

        return labels
        