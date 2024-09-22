import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader



class CustomDataset(Dataset):
    def __init__(self, full_dose_path, quarter_dose_path):
        self.full_dose_path = full_dose_path
        self.quarter_dose_path = quarter_dose_path
        self.length = len(self.quarter_dose_path)

    def __len__(self):
        return self.length

    def norm(self, x):
        if np.amax(x) > 0:
            x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        return x

    def __getitem__(self, data_id):
        img_path_f = self.full_dose_path[data_id]
        img_path_q = self.quarter_dose_path[data_id]

        img_f = np.array(Image.open(img_path_f), dtype=np.float32) / 255
        img_q = np.array(Image.open(img_path_q), dtype=np.float32) / 255



        return self.norm(img_q.reshape(1, 256, 256)), self.norm(img_f.reshape(1, 256, 256))
    
    
    
def create_loaders_mix(full_dose_path , quarter_dose_path , dataset_ratio , train_ratio , batch_size):  #used in train
    
    full_dose_path_list = []
    quarter_dose_path_list = []
    
    for mm in os.listdir(full_dose_path):
        mm_path_fdose = os.path.join(full_dose_path, mm)
        mm_path_qdose = os.path.join(quarter_dose_path, mm)
        for k_type in os.listdir(mm_path_fdose):
            kernel_path_fdose = os.path.join(mm_path_fdose, k_type)
            kernel_path_qdose = os.path.join(mm_path_qdose, k_type)
            for L in os.listdir(kernel_path_fdose):
                L_path_fdose = os.path.join(kernel_path_fdose, L)
                L_path_qdose = os.path.join(kernel_path_qdose, L)
                temp_img_list_f = []
                temp_img_list_q = []
                for im in os.listdir(L_path_fdose):
                    im_dir_f = os.path.join(L_path_fdose, im)
                    temp_img_list_f.append(im_dir_f)
                for im in os.listdir(L_path_qdose):
                    im_dir_q = os.path.join(L_path_qdose, im)
                    temp_img_list_q.append(im_dir_q)
    
                for im_name_f in temp_img_list_f:
                    for im_name_q in temp_img_list_q:
                        if im_name_f.split(".")[3] == im_name_q.split(".")[3]:
                            full_dose_path_list.append(im_name_f)
                            quarter_dose_path_list.append(im_name_q)
    
    
    
    perm = np.random.permutation(len(full_dose_path_list))
    number_of_image_pairs = int(dataset_ratio * len(perm))
    
    shuffled_f = [full_dose_path_list[i] for i in perm][:number_of_image_pairs]
    shuffled_q = [quarter_dose_path_list[i] for i in perm][:number_of_image_pairs]
    
    print("Lists of paths are done")
    
    
    
    
    
    train_len = int(len(shuffled_f) * train_ratio)

    
    train_data = CustomDataset(shuffled_f[:train_len], shuffled_q[:train_len])
    val_data = CustomDataset(shuffled_f[train_len:], shuffled_q[train_len:])   
    
    trainloader = DataLoader(train_data, batch_size=batch_size)
    validloader = DataLoader(val_data, batch_size=1)
    
    return trainloader , validloader


def create_loaders_seperate(full_dose_path , quarter_dose_path , dataset_ratio , train_ratio , batch_size , mm_type): #used in test
    
    full_dose_path_list = []
    quarter_dose_path_list = []
    
    for mm in os.listdir(full_dose_path):
        if mm == mm_type:
            mm_path_fdose = os.path.join(full_dose_path, mm)
            mm_path_qdose = os.path.join(quarter_dose_path, mm)
            for k_type in os.listdir(mm_path_fdose):
                kernel_path_fdose = os.path.join(mm_path_fdose, k_type)
                kernel_path_qdose = os.path.join(mm_path_qdose, k_type)
                for L in os.listdir(kernel_path_fdose):
                    L_path_fdose = os.path.join(kernel_path_fdose, L)
                    L_path_qdose = os.path.join(kernel_path_qdose, L)
                    temp_img_list_f = []
                    temp_img_list_q = []
                    for im in os.listdir(L_path_fdose):
                        im_dir_f = os.path.join(L_path_fdose, im)
                        temp_img_list_f.append(im_dir_f)
                    for im in os.listdir(L_path_qdose):
                        im_dir_q = os.path.join(L_path_qdose, im)
                        temp_img_list_q.append(im_dir_q)

                    for im_name_f in temp_img_list_f:
                        for im_name_q in temp_img_list_q:
                            if im_name_f.split(".")[3] == im_name_q.split(".")[3]:
                                full_dose_path_list.append(im_name_f)
                                quarter_dose_path_list.append(im_name_q)
    
    
    
    perm = np.random.permutation(len(full_dose_path_list))
    number_of_image_pairs = int(dataset_ratio * len(perm))
    
    shuffled_f = [full_dose_path_list[i] for i in perm][:number_of_image_pairs]
    shuffled_q = [quarter_dose_path_list[i] for i in perm][:number_of_image_pairs]
    
    print("Lists of paths are done")
    
    
    
    
    train_len = int(len(shuffled_f) * train_ratio)

    
    train_data = CustomDataset(shuffled_f[:train_len], shuffled_q[:train_len])
    val_data = CustomDataset(shuffled_f[train_len:], shuffled_q[train_len:])   
    
    trainloader = DataLoader(train_data, batch_size=batch_size)
    validloader = DataLoader(val_data, batch_size=1)
    
    return trainloader , validloader