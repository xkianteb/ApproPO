from ApproPO.olo import OGD
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import namedtuple

class ProjectionOracle:
    def __init__(self, dim=None, proj=None, args=None):
        self.dim = dim
        self.proj = proj
        self.lr = args.proj_lr
        self.mx_size = args.mx_size
        self.olo = OGD(dim+1, self.proj_decision_set, self.lr, args=args)
        self.diversity = args.diversity

    def g(self, p=None, alpha=None):
        p_on_plane = p[:-1]
        q = self.proj((self.mx_size / alpha)*p_on_plane)
        q *= (alpha/self.mx_size)
        q = np.append(q, alpha)
        return np.linalg.norm(q-p), q

    def proj_cone(self, p):
        epsilon = 1e-6
        left = 0.0
        right = 100
        while np.abs(right-left) > epsilon:
            #m1 = (2*left+right)/3.0
            #m2 = (left+2*right)/3.0
            m1 = left + (right - left)/3
            m2 = right - (right - left)/3
            if self.g(p, m1)[0] < self.g(p, m2)[0]:
                right = m2
            else:
                left = m1
        return self.g(p, (left+right)/2.0)[1]


    def proj_polar_cone(self, p):
        q = self.proj_cone(p)
        return p-q

    def proj_decision_set(self, p):
        for i in range(20): #TODO: is needed?
            p = self.proj_polar_cone(p)
            if np.linalg.norm(p) > 1:
                p = p / np.linalg.norm(p)
        return p

    def get_theta(self):
        return self.olo.get_theta()

    def update(self, expected_return):
        if not self.diversity:
            expected_return[2:] = 0
        expected_return = np.append(expected_return, self.mx_size)
        loss_vector = -expected_return
        self.olo.step_linear(loss_vector)

    def reset(self):
        self.olo.reset()

