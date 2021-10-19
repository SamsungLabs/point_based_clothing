import torch
from torch.utils.data import DataLoader
from utils.utils import load_module


class Dataloader:
    def __init__(self, dataset_name):
        self.dataset = self.find_definition(dataset_name)

    def find_definition(self, dataset_name):

        m = load_module('dataloaders', dataset_name)
        print(m)
        return m.__dict__['Dataset']

    def get_args(self, parser):
        parser.add('--num_workers', type=int, default=4, help='Number of data loading workers.')
        parser.add('--batch_size', type=int, default=64, help='Batch size')

        return self.dataset.get_args(parser)

    def get_dataloader(self, args, part, phase):
        if hasattr(self.dataset, 'get_dataloader'):
            return self.dataset.get_dataloader(args, part)
        else:
            dataset = self.dataset.get_dataset(args, part)

            return DataLoader(
                dataset,
                batch_size=args.batch_size if phase == 'train' else 1,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True if phase == 'train' else False,
                shuffle=True if part == 'train' else False)