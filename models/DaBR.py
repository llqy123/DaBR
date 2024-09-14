import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class DaBR(Model):
    def __init__(self, config):
        super(DaBR, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.config.entTotal, 4 * self.config.hidden_size)  # vectorized quaternion
        self.rel_embeddings = nn.Embedding(self.config.relTotal, 4 * self.config.hidden_size)
        self.Dr = nn.Embedding(self.config.relTotal, 4 * self.config.hidden_size)
        self.para = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.criterion = nn.Softplus()
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.Dr.weight.data)

    @staticmethod
    def normalization(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
        size = quaternion.size(split_dim) // 4
        quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
        quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))  # quaternion / norm
        quaternion = quaternion.reshape(-1, 4 * size)
        return quaternion

    @staticmethod
    def make_wise_quaternion(quaternion):  # for vector * vector quaternion element-wise multiplication
        if len(quaternion.size()) == 1:
            quaternion = quaternion.unsqueeze(0)
        size = quaternion.size(1) // 4
        r, i, j, k = torch.split(quaternion, size, dim=1)
        r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3 --> bs x 4dim
        i2 = torch.cat([i, r, -k, j], dim=1)  # 1, 0, 3, 2
        j2 = torch.cat([j, k, r, -i], dim=1)  # 2, 3, 0, 1
        k2 = torch.cat([k, -j, i, r], dim=1)  # 3, 2, 1, 0
        return r2, i2, j2, k2

    @staticmethod
    def get_quaternion_wise_mul(quaternion):
        size = quaternion.size(1) // 4
        quaternion = quaternion.view(-1, 4, size)
        quaternion = torch.sum(quaternion, 1)
        return quaternion

    @staticmethod
    def vec_vec_wise_multiplication(q, p):  # vector * vector
        normalized_p = DaBR.normalization(p)  # bs x 4dim
        q_r, q_i, q_j, q_k = DaBR.make_wise_quaternion(q)  # bs x 4dim

        qp_r = DaBR.get_quaternion_wise_mul(q_r * normalized_p)  # qrpr−qipi−qjpj−qkpk
        qp_i = DaBR.get_quaternion_wise_mul(q_i * normalized_p)  # qipr+qrpi−qkpj+qjpk
        qp_j = DaBR.get_quaternion_wise_mul(q_j * normalized_p)  # qjpr+qkpi+qrpj−qipk
        qp_k = DaBR.get_quaternion_wise_mul(q_k * normalized_p)  # qkpr−qjpi+qipj+qrpk

        return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)

    @staticmethod
    def get_inv(quaternion):
        q_r, q_i, q_j, q_k = torch.chunk(quaternion, 4, dim=1)
        quaternion_norm = q_r ** 2 + q_i ** 2 + q_j ** 2 + q_k ** 2
        r_inv = torch.cat([q_r / quaternion_norm, -q_i / quaternion_norm, -q_j / quaternion_norm, -q_k / quaternion_norm], dim=1)
        return r_inv

    @staticmethod
    def _calc(h, r, t, dr, para):
        #semantic
        hr = DaBR.vec_vec_wise_multiplication(h, r)
        r_inv = DaBR.get_inv(r)
        tr = DaBR.vec_vec_wise_multiplication(t, r_inv)
        score_s = hr * tr
        #distance
        hrt = h + dr - t
        s_d, x_d, y_d, z_d = torch.chunk(hrt, 4, dim=1)
        score_d = s_d + x_d + y_d + z_d
        return -torch.sum(score_s, -1) - para * torch.norm(score_d, p=1, dim=-1)

    @staticmethod
    def regularization(quaternion):  # vectorized quaternion bs x 4dim
        size = quaternion.size(1) // 4
        r, i, j, k = torch.split(quaternion, size, dim=1)
        return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)


    def loss(self, score, regul1, regul2):
        return torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul1 + self.config.lmbda_two * regul2

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        dr = self.Dr(self.batch_r)
        para = self.para
        score = self._calc(h, r, t, dr, para)
        regul1 = self.regularization(h) + self.regularization(t)
        regul2 = self.regularization(r) + self.regularization(dr)
        return self.loss(score, regul1, regul2)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        dr = self.Dr(self.batch_r)
        para = self.para
        score = self._calc(h, r, t, dr, para)

        return score.cpu().data.numpy()
