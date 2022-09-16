"""Extracting Content-Aware Perceptual Features using Pre-Trained Image Classification Models (e.g., ResNet-50)"""
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2019/11/8

from argparse import ArgumentParser
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
import time


class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height, self.width, inputdict={'-pix_fmt':'yuvj420p'})
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        print('video_width: {} video_height: {}'.format(video_width, video_height))
        transformed_video = torch.zeros([video_length, video_channel,  video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            # frame.show()
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score}

        return sample


class CNNModel(torch.nn.Module):
    """Modified CNN models for feature extraction"""
    def __init__(self, model='ResNet-50'):
        super(CNNModel, self).__init__()
        if model == 'AlexNet':
            print("use AlexNet")
            self.features = nn.Sequential(*list(models.alexnet(pretrained=True).children())[:-2])
        elif model == 'ResNet-152':
            print("use ResNet-152")
            self.features = nn.Sequential(*list(models.resnet152(pretrained=True).children())[:-2])
        elif model == 'ResNeXt-101-32x8d':
            print("use ResNetXt-101-32x8d")
            self.features = nn.Sequential(*list(models.resnext101_32x8d(pretrained=True).children())[:-2])
        elif model == 'Wide ResNet-101-2':
            print("use Wide ResNet-101-2")
            self.features = nn.Sequential(*list(models.wide_resnet101_2(pretrained=True).children())[:-2])
        else:
            print("use default ResNet-50")
            self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])

    def forward(self, x):
        x = self.features(x)
        features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
        features_std = global_std_pool2d(x)
        return features_mean, features_std
        # # features@: 7->res5c
        # for ii, model in enumerate(self.features):
        #     x = model(x)
        #     if ii == 7:
        #         features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
        #         features_std = global_std_pool2d(x)
        #         return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, frame_batch_size=64, model='ResNet-50', device='cuda'):
    """feature extraction"""
    extractor = CNNModel(model=model).to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].to(device)
            features_mean, features_std = extractor(batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        features_mean, features_std = extractor(last_batch)
        output1 = torch.cat((output1, features_mean), 0)
        output2 = torch.cat((output2, features_std), 0)
        output = torch.cat((output1, output2), 1).squeeze()

    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using pre-trained models')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='CVD2014', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='ResNet-50', type=str,
                        help='which pre-trained model used (default: ResNet-50)')
    parser.add_argument('--frame_batch_size', type=int, default=8,
                        help='frame batch size for feature extraction (default: 8)')

    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')

    parser.add_argument("--ith", type=int, default=0, help='start frame id')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        videos_dir = 'KoNViD-1k/'  # videos dir, e.g., ln -s /home/ldq/Downloads/KoNViD-1k/ KoNViD-1k
        features_dir = 'CNN_features_KoNViD-1k/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        videos_dir = 'CVD2014/' # ln -s /media/ldq/Research/Data/CVD2014/ CVD2014
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        videos_dir = 'LIVE-Qualcomm/' # ln -s /media/ldq/Others/Data/12.LIVE-Qualcomm\ Mobile\ In-Capture\ Video\ Quality\ Database/ LIVE-Qualcomm
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'
    if args.database == 'LIVE-VQC':
        videos_dir = 'LIVE-VQC/' #  /media/ldq/Others/Data/LIVE\ Video\ Quality\ Challenge\ \(VQC\)\ Database/Video LIVE-VQC
        features_dir = 'CNN_features_LIVE-VQC/'
        datainfo = 'data/LIVE-VQCinfo.mat'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])
    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)

    max_len = 0
    # extract feature on LIVE-Qualcomm using AlexNet will cause the error of "cannot allocate memory"
    # One way to solve the problem is to move the for loop to bash.
    """
    for ((i=0; i<208; i++)); do
        CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --ith=$i --model=AlexNet --database=LIVE-Qualcomm
    done
    """
    for i in range(args.ith, len(dataset)): # range(args.ith, args.ith+1): # 
        start = time.time()
        current_data = dataset[i]
        print('Video {}: length {}'.format(i, current_data['video'].shape[0]))
        if max_len < current_data['video'].shape[0]:
            max_len = current_data['video'].shape[0]
        features = get_features(current_data['video'], args.frame_batch_size, args.model, device)
        np.save(features_dir + str(i) + '_' + args.model +'_last_conv', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_data['score'])
        end = time.time()
        print('{} seconds'.format(end-start))
    print(max_len)
