from torch.utils.data import DataLoader

from dataloaders import internet_images
from dataloaders.loaders.internet_images_loader import Loader

from utils.utils import load_module


def make_dataloader(name, data_root, rgb_dir, segm_dir, smpl_dir, datalist, smpl_model_path, 
                    train=True, batch_size=1, image_size=512, max_iter=1):
    '''
    Args:
        name (`str`): name of the dataset. 
         We recommend to create at least two files for each dataset:
         - `dataloaders/<name>.py` with the `ClothDataset` class definition for this dataset
         - `dataloaders/loaders/<name>_loader.py`  with the `Loader` class definition for this dataset
         - [optional] place your dataset in `samples/<name>/` folder.
    '''
    
    m_loader = load_module('dataloaders.loaders', name+'_loader')
    m_dataloader = load_module('dataloaders', name)
    
    loader = m_loader.Loader(data_root, rgb_dir, segm_dir, smpl_dir, image_size, smpl_model_path=smpl_model_path)
    dataset = m_dataloader.ClothDataset(loader, datalist)
    
    if train:
        if isinstance(datalist, list):
            dataset.datalist = dataset.datalist * (max_iter // len(datalist)) * batch_size
        else:  # pd.DataFrame
            dataset.datalist = datalist.loc[datalist.index.repeat((max_iter // len(datalist)) * batch_size)]
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return dataloader


def make_dataloaders(name, data_root, rgb_dir, segm_dir, smpl_dir, datalist, smpl_model_path, batch_size, max_iter=1):
    '''
    Args:
        name (`str`): name of the dataset. 
         We recommend to create at least two files for each dataset:
         - `dataloaders/<name>.py` with the `ClothDataset` class definition for this dataset
         - `dataloaders/loaders/<name>_loader.py`  with the `Loader` class definition for this dataset
         - [optional] place your dataset in `samples/<name>/` folder.
    '''
    
    dataloader = make_dataloader(name, data_root, rgb_dir, segm_dir, smpl_dir, datalist, smpl_model_path, 
                                 train=True, batch_size=batch_size, max_iter=max_iter)
    print('dataloader', len(dataloader))
    
    dataloader_val = make_dataloader(name, data_root, rgb_dir, segm_dir, smpl_dir, datalist, smpl_model_path, 
                                     train=False, batch_size=1, max_iter=max_iter)
    print('dataloader_val', len(dataloader_val))
    
    return dataloader, dataloader_val
