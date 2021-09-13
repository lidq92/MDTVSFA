import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


class VQADataset(Dataset):
    def __init__(self, args, datasets, status='train'):
        self.status = status
        self.datasets = datasets
        self.crop_length = args.crop_length

        max_len = dict()
        self.M = dict()
        self.m = dict()
        self.scale = dict()
        self.index = dict()

        for dataset in datasets:
            Info = h5py.File(args.data_info[dataset], 'r')
            max_len[dataset] = int(Info['max_len'][0])

            self.M[dataset] = Info['scores'][0, :].max()
            self.m[dataset] = Info['scores'][0, :].min()
            self.scale[dataset] = self.M[dataset] - self.m[dataset]

            index = Info['index']
            index = index[:, args.exp_id % index.shape[1]]
            ref_ids = Info['ref_ids'][0, :]
            if status == 'train':
                index = index[0:int(args.train_proportion * args.train_ratio * len(index))]
            elif status == 'val':
                index = index[int(args.train_ratio * len(index)):int((0.5 + args.train_ratio / 2) * len(index))]
            elif status == 'test':
                    index = index[int((0.5 + args.train_ratio / 2) * len(index)):len(index)]
            self.index[dataset] = []
            for i in range(len(ref_ids)):
                if ref_ids[i] in index:
                    self.index[dataset].append(i)
            print("# {} images from {}: {}".format(status, dataset, len(self.index[dataset])))
            print("Ref Index: ")
            print(index.astype(int))

        max_len_all = max(max_len.values())
        self.features, self.length, self.label, self.KCL, self.N = dict(), dict(), dict(), dict(), dict()
        for dataset in datasets:
            N = len(self.index[dataset])
            self.N[dataset] = N
            self.features[dataset] = np.zeros((N, max_len_all, args.feat_dim), dtype=np.float32)
            self.length[dataset] = np.zeros(N, dtype=np.int)
            self.label[dataset] = np.zeros((N, 1), dtype=np.float32)
            self.KCL[dataset] = []
            for i in range(N):
                features = np.load(args.features_dir[dataset] + str(self.index[dataset][i]) + '_' + args.feature_extractor +'_last_conv.npy')
                self.length[dataset][i] = features.shape[0]
                self.features[dataset][i, :features.shape[0], :] = features
                mos = np.load(args.features_dir[dataset] + str(self.index[dataset][i]) + '_score.npy')  #
                self.label[dataset][i] = mos
                self.KCL[dataset].append(dataset)

    def __len__(self):
        return max(self.N.values())

    def __getitem__(self, idx):
        data = [(self.features[dataset][idx % self.N[dataset]],
                 self.length[dataset][idx % self.N[dataset]],
                 self.KCL[dataset][idx % self.N[dataset]]) for dataset in self.datasets]
        label = [self.label[dataset][idx % self.N[dataset]] for dataset in self.datasets]
        return data, label


def get_data_loaders(args):
    """ Prepare the train-val-test data
    :param args: related arguments
    :return: train_loader, val_loader, test_loader
    """
    train_dataset = VQADataset(args, args.datasets['train'], 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               drop_last=True)  #

    scale = train_dataset.scale
    m = train_dataset.m

    val_loader, test_loader = dict(), dict()
    for dataset in args.datasets['val']:
        val_dataset = VQADataset(args, [dataset], 'val')
        val_loader[dataset] = torch.utils.data.DataLoader(val_dataset)

    for dataset in args.datasets['test']:
        test_dataset = VQADataset(args, [dataset], 'test')
        if dataset not in args.datasets['train']:
            scale[dataset] = test_dataset.scale[dataset]
            m[dataset] = test_dataset.m[dataset]
        test_loader[dataset] = torch.utils.data.DataLoader(test_dataset)

    return train_loader, val_loader, test_loader, scale, m
