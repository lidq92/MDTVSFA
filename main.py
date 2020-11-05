# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2019/11/8
#
# source activate reproducibleresearch
# tensorboard --logdir=runs --port=6006

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from VQAdataset import get_data_loaders
from VQAmodel import VQAModel
from VQAloss import VQALoss
from VQAperformance import VQAPerformance
from tensorboardX import SummaryWriter
import datetime
import os
import numpy as np
import random
from argparse import ArgumentParser


def writer_add_scalar(writer, status, dataset, scalars, iter):
    writer.add_scalar("{}/{}/SROCC".format(status, dataset), scalars['SROCC'], iter)
    writer.add_scalar("{}/{}/KROCC".format(status, dataset), scalars['KROCC'], iter)
    writer.add_scalar("{}/{}/PLCC".format(status, dataset), scalars['PLCC'], iter)
    writer.add_scalar("{}/{}/RMSE".format(status, dataset), scalars['RMSE'], iter)


def run(args):
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, scale, m = get_data_loaders(args)
    model = VQAModel(scale, m, args.simple_linear_scale).to(device)  #
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    loss_func = VQALoss([scale[dataset] for dataset in args.datasets['train']], args.loss, [m[dataset] for dataset in args.datasets['train']])
    trainer = create_supervised_trainer(model, optimizer, loss_func, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'VQA_performance': VQAPerformance()}, device=device)

    if args.inference:
        model.load_state_dict(torch.load(args.trained_model_file))
        performance = dict()
        for dataset in args.datasets['test']:
            evaluator.run(test_loader[dataset])
            performance[dataset] = evaluator.state.metrics['VQA_performance']
            print('{}, SROCC: {}'.format(dataset, performance[dataset]['SROCC']))
        np.save(args.save_result_file, performance)
        return

    writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'
                           .format(args.log_dir, args.exp_id, args.model, args.feature_extractor, args.loss, args.train_proportion, args.datasets['train'],
                                   args.lr, args.batch_size, args.epochs,
                                   datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

    global best_val_criterion, best_epoch
    best_val_criterion, best_epoch = -100, -1  # larger, better, e.g., SROCC/KROCC/PLCC

    @trainer.on(Events.ITERATION_COMPLETED)
    def iter_event_function(engine):
        writer.add_scalar("train/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_event_function(engine):
        val_criterion = 0
        for dataset in args.datasets['val']:
            evaluator.run(val_loader[dataset])
            performance = evaluator.state.metrics['VQA_performance']
            writer_add_scalar(writer, 'val', dataset, performance, engine.state.epoch)
            if dataset in args.datasets['train']:
                val_criterion += performance['SROCC']

        for dataset in args.datasets['test']:
            evaluator.run(test_loader[dataset])
            performance = evaluator.state.metrics['VQA_performance']
            writer_add_scalar(writer, 'test', dataset, performance, engine.state.epoch)

        global best_val_criterion, best_epoch
        if val_criterion > best_val_criterion:
            torch.save(model.state_dict(), args.trained_model_file)
            best_val_criterion = val_criterion
            best_epoch = engine.state.epoch
            print('Save current best model @best_val_criterion: {} @epoch: {}'.format(best_val_criterion, best_epoch))

        scheduler.step(engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        print('best epoch: {}'.format(best_epoch))
        model.load_state_dict(torch.load(args.trained_model_file))
        performance = dict()
        for dataset in args.datasets['test']:
            evaluator.run(test_loader[dataset])
            performance[dataset] = evaluator.state.metrics['VQA_performance']
            print('{}, SROCC: {}'.format(dataset, performance[dataset]['SROCC']))
        np.save(args.save_result_file, performance)

    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    parser = ArgumentParser(description='Mixed Dataset Training for Quality Assessment of In-the-Wild Videos')
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

    parser.add_argument('--trained_datasets', nargs='+', type=str, default=['K', 'C', 'L', 'N'],
                        help="trained datasets (default: ['K', 'C', 'L', 'N'])")

    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--crop_length', type=int, default=180,
                        help='Crop video length (<=max_len=1202, default: 180)')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='train ratio (default: 0.6)')
    parser.add_argument('--train_proportion', type=float, default=6,
                        help='the number of proportions (#total 6) used in the training set (default: 6)')

    parser.add_argument("--log_dir", type=str, default="runs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    parser.add_argument('--inference', action='store_true',
                        help='Inference?')
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

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    args.trained_model_file = 'checkpoints/{}-{}-{}-{}-{}-{}-{}-{}-EXP{}'.format(args.model, args.feature_extractor, args.loss, args.train_proportion, args.datasets['train'], args.lr, args.batch_size, args.epochs, args.exp_id)
    if not os.path.exists('results'):
        os.makedirs('results')
    args.save_result_file = 'results/{}-{}-{}-{}-{}-{}-{}-{}-EXP{}'.format(args.model, args.feature_extractor, args.loss, args.train_proportion, args.datasets['train'], args.lr, args.batch_size, args.epochs, args.exp_id)
    print(args)
    run(args)
