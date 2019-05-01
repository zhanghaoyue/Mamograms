"""
Utility functions.
"""

import torch.utils.data as data
import torch
import numpy as np


class MMDataset(data.Dataset):
    """
    Pytorch Dataset class for TSS. Assume the data is already pre-loaded into memory
    """

    def __init__(self, data, patients, transform=None, datasetType='Train'):
        """
        :param data: the data dict that contains imgs and label for every patients.
                     each data has: data['imgs'], data['imgs_precise'], data['label']
                     e.g., data['imgs'] has dimension of x,y,z,type_of_img(e.g., cbf)
        :param patients: patients to be used
        :param transform: the transform for the data
        :param datasetType: the type of the data, train or test. 
        :param is_roi_filtered: whether to use precise or complete imags
        :param num_augments: number of augmented samples
        """        
        self.data = data
        self.patients = patients
        self.transform = transform       
        self.datasetType = datasetType

        self.total_patients = len(self.patients)
    
        
    def __getitem__(self, index):
        """
        For pytorch it is expected we use dataLoader to load a batch of samples. Typically, the 2D
        img data in a batch should have a dimension of batch_num  x channel x height x width. Here, 
        we have 3D img data in which the original dim is height x width x depth x imagetype(e.g., cbf).
        We need to rearrage the image such that we have an output of 
        
        imagetype x depth x height x width <-- We treat imagetype as color channel
        
        :param index: the index of the item        
        """
        if self.datasetType == 'Train':
            actual_index = index % self.total_patients

            idx = self.patients[actual_index]   
            label = self.data[idx]['label']
        else:        
            idx = self.patients[index]   
            label = self.data[idx]['label']

        # define what type of images to be used        
        img = self.data[idx]['imgs'] 

        # transform data first
        if self.transform:
            img = self.transform(img)

        
        # create sample
        sample = {'idx':idx,'img': img, 'label': label}
               
        #return sample
        return img, label

    def __len__(self):
        return self.total_patients
