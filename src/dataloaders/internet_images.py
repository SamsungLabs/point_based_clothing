import os
import pickle


class ClothDataset:
    def __init__(self, loader, datalist):
        self.datalist = datalist
        self.loader = loader

    def get_sample(self, sample):
        data_dict, target_dict = self.loader.load_sample(sample)

        return data_dict, target_dict

    def __getitem__(self, item):
        sample = self.datalist[item]

        data_dict, target_dict = self.get_sample(sample)
        data_dict['index'] = item
        data_dict['fid'] = sample

        return data_dict, target_dict

    def __len__(self):
        return len(self.datalist)
