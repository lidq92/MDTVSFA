# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2019/11/8
#

import torch
from torch.utils.data import Dataset
from ignite.engine import create_supervised_evaluator
from VQAmodel import VQAModel
from VQAloss import VQALoss
from VQAperformance import VQAPerformance
import datetime
import os
import numpy as np
import random
from argparse import ArgumentParser
import h5py


class VQADataset(Dataset):
    def __init__(self, args, datasets):
        self.datasets = datasets

        self.index = dict()
        max_len = dict()

        for dataset in datasets:
            Info = h5py.File(args.data_info[dataset], 'r')
            max_len[dataset] = int(Info['max_len'][0])
            index = Info['index']
            index = index[:, args.exp_id % index.shape[1]]
            ref_ids = Info['ref_ids'][0, :]
            self.index[dataset] = []
            for i in range(len(ref_ids)):
                if ref_ids[i] in index:
                    self.index[dataset].append(i)

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


def run(args):
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    test_loader = dict()
    for dataset in args.cross_datasets:
        test_dataset = VQADataset(args, [dataset])
        test_loader[dataset] = torch.utils.data.DataLoader(test_dataset)

    model = VQAModel(simple_linear_scale=args.simple_linear_scale).to(device)  #
    model.load_state_dict(torch.load(args.trained_model_file))

    evaluator = create_supervised_evaluator(model, metrics={'VQA_performance': VQAPerformance()}, device=device)

    performance = dict()
    for dataset in args.cross_datasets:
        evaluator.run(test_loader[dataset])
        performance[dataset] = evaluator.state.metrics['VQA_performance']
        print('{}, SROCC: {}'.format(dataset, performance[dataset]['SROCC']))
    np.save(args.save_result_file, performance)


if __name__ == "__main__":
    parser = ArgumentParser(description='MDTVSFA Cross-dataset evaluation')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument('--model', default='MDTVSFA', type=str,
                        help='model name (default: MDTVSFA)')
    parser.add_argument('--loss', default='mixed', type=str,
                        help='loss type (default: mixed)')    
    parser.add_argument('--feature_extractor', default='ResNet-50', type=str,
                        help='feature_extractor backbone (default: ResNet-50)')
    # parser.add_argument('--feat_dim', type=int, default=4096,
    #                     help='feature dimension (default: 4096)')

    parser.add_argument('--trained_datasets', nargs='+', type=str, default=['K'],
                        help="trained datasets (default: ['K'])")

    parser.add_argument('--cross_datasets', nargs='+', type=str, default=['C', 'L', 'N'],
                        help="cross datasets (default: ['C', 'L', 'N'])")

    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--train_proportion', type=float, default=6,
                        help='the number of proportions (#total 6) used in the training set (default: 6)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()
    args.train_proportion /= 6
    if args.feature_extractor == 'AlexNet':
        args.feat_dim = 256 * 2
    else:
        args.feat_dim = 2048 * 2


    args.simple_linear_scale = False  #
    if 'naive' in args.loss:
        args.simple_linear_scale = True  #

    args.decay_interval = int(args.epochs / 20)
    args.decay_ratio = 0.8

    args.datasets = {'train': args.trained_datasets,
                     'val': args.trained_datasets,
                     'test': ['K', 'C', 'L', 'N']}
    args.features_dir = {'K': 'CNN_features_KoNViD-1k/',
                         'C': 'CNN_features_CVD2014/',
                         'L': 'CNN_features_LIVE-Qualcomm/',
                         'N': 'CNN_features_LIVE-VQC/'}
    args.data_info = {'K': 'data/KoNViD-1kinfo.mat',
                      'C': 'data/CVD2014info.mat',
                      'L': 'data/LIVE-Qualcomminfo.mat',
                      'N': 'data/LIVE-VQCinfo.mat'}

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    args.trained_model_file = 'checkpoints/{}-{}-{}-{}-{}-{}-{}-{}-EXP{}'.format(args.model, args.feature_extractor, args.loss, args.train_proportion, args.trained_datasets, args.lr, args.batch_size, args.epochs, args.exp_id)
    if not os.path.exists('results'):
        os.makedirs('results')
    args.save_result_file = 'results/cross-dataset-{}-{}-{}-{}-{}-{}-{}-{}-EXP{}'.format(args.model, args.feature_extractor, args.loss, args.train_proportion, args.trained_datasets, args.lr, args.batch_size, args.epochs, args.exp_id)
    print(args)
    run(args)
