import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class VQAModel(nn.Module):
    def __init__(self, scale={'K': 1, 'C': 1, 'L': 1, 'N': 1}, m={'K': 0, 'C': 0, 'L': 0, 'N': 0}, 
                 simple_linear_scale=False, input_size=4096, reduced_size=128, hidden_size=32):
        super(VQAModel, self).__init__()
        self.hidden_size = hidden_size
        mapping_datasets = scale.keys()

        self.dimemsion_reduction = nn.Linear(input_size, reduced_size)
        self.feature_aggregation = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.regression = nn.Linear(hidden_size, 1)
        self.bound = nn.Sigmoid()
        self.nlm = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid(), nn.Linear(1, 1))  # 4 parameters
        # self.nlm = nn.Sequential(nn.Sequential(nn.Linear(1, 1), nn.Sigmoid(), nn.Linear(1, 1, bias=False)),
        #                          nn.Linear(1, 1))  # 5 parameters
        self.lm = nn.Sequential(OrderedDict([(dataset, nn.Linear(1, 1)) for dataset in mapping_datasets]))

        torch.nn.init.constant_(self.nlm[0].weight, 2*np.sqrt(3))
        torch.nn.init.constant_(self.nlm[0].bias, -np.sqrt(3))
        torch.nn.init.constant_(self.nlm[2].weight, 1)
        torch.nn.init.constant_(self.nlm[2].bias, 0)
        for p in self.nlm[2].parameters():
            p.requires_grad = False
        for d, dataset in enumerate(mapping_datasets):
            torch.nn.init.constant_(self.lm._modules[dataset].weight, scale[dataset])
            torch.nn.init.constant_(self.lm._modules[dataset].bias, m[dataset])


        # torch.nn.init.constant_(self.nlm[0][0].weight, 2*np.sqrt(3))
        # torch.nn.init.constant_(self.nlm[0][0].bias, -np.sqrt(3))
        # torch.nn.init.constant_(self.nlm[0][2].weight, 0)

        # torch.nn.init.constant_(self.nlm[1].weight, 1)
        # torch.nn.init.constant_(self.nlm[1].bias, 0)
        # for d, dataset in enumerate(mapping_datasets):
        #     torch.nn.init.constant_(self.lm._modules[dataset].weight, scale[dataset])
        #     torch.nn.init.constant_(self.lm._modules[dataset].bias, m[dataset])
            
        # for d, dataset in enumerate(mapping_datasets):
        #     if d == 0:
        #         dataset0 = dataset
        #         torch.nn.init.constant_(self.nlm[1].weight, scale[dataset0])
        #         torch.nn.init.constant_(self.nlm[1].bias, m[dataset0])
        #         torch.nn.init.constant_(self.lm._modules[dataset0].weight, 1)
        #         torch.nn.init.constant_(self.lm._modules[dataset0].bias, 0)
        #         for p in self.lm._modules[dataset0].parameters():
        #             p.requires_grad = False
        #     else:
        #         torch.nn.init.constant_(self.lm._modules[dataset].weight, scale[dataset] / scale[dataset0])
        #         torch.nn.init.constant_(self.lm._modules[dataset].bias,
        #                                 m[dataset] - m[dataset0] * scale[dataset] / scale[dataset0])

        if simple_linear_scale:
            for p in self.lm.parameters():
                p.requires_grad = False

    def forward(self, input):
        relative_score, mapped_score, aligned_score = [], [], []
        for d, (x, x_len, KCL) in enumerate(input):
            x = self.dimemsion_reduction(x)  # dimension reduction
            x, _ = self.feature_aggregation(x, self._get_initial_state(x.size(0), x.device))
            q = self.regression(x)  # frame quality
            relative_score.append(torch.zeros_like(q[:, 0]))  #
            mapped_score.append(torch.zeros_like(q[:, 0]))  #
            aligned_score.append(torch.zeros_like(q[:, 0]))  #
            for i in range(q.shape[0]):  #
                relative_score[d][i] = self._sitp(q[i, :x_len[i].item()])  # video overall quality
            relative_score[d] = self.bound(relative_score[d])
            # mapped_score[d] = relative_score[d] # The nonlinear mapping module is embedded into the RQA.
            mapped_score[d] = self.nlm(relative_score[d]) # 4 parameters
            # mapped_score[d] = self.nlm[0](relative_score[d]) + self.nlm[1](relative_score[d]) # 5 parameters
            for i in range(q.shape[0]):
                aligned_score[d][i] = self.lm._modules[KCL[i]](mapped_score[d][i])

        return relative_score, mapped_score, aligned_score

    def _sitp(self, q, tau=12, beta=0.5):
        """subjectively-inspired temporal pooling"""
        q = torch.unsqueeze(torch.t(q), 0)
        qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
        qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
        l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
        m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
        n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
        m = m / n
        q_hat = beta * m + (1 - beta) * l
        return torch.mean(q_hat)

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0
