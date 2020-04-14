#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, F, H, Q, R, B=None, P=None, x0=None):
        self.n = F.shape[1]
        self.m = F.shape[1]

        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = 0 if B is None else B
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

        self.sigma00 = []
        self.sigma01 = []
        self.sigma11 = []
        self.sigma10 = []

    def predict(self, u=0, Q=None):
        self.Q = self.Q if Q == None else Q
        
        self.x = dot(self.F, self.x) + dot(self.B, u)
        self.P = dot(dot(self.F, self.P), self.F.T) + self.Q

    def correct(self, z, R=None):
        self.R = self.R if R == None else R

        residual = z - dot(self.H, self.x)
        S = dot(dot(self.H, self.P), self.H.T) + self.R
        K = dot(dot(self.P, self.H.T), inv(S))

        self.x = self.x + dot(K, residual)
        self.P = dot(np.eye(self.n) - dot(K, self.H), self.P)
        self.sigma00.append(self.P[0][0])
        self.sigma01.append(self.P[0][1])
        self.sigma11.append(self.P[1][1])
        self.sigma10.append(self.P[1][0])

if __name__ == "__main__":
    dt = 1.0/60
    F = np.array([[1, dt], [0, 1]])
    H = np.array([1, 0]).reshape(1, 2)
    Q = np.array([[0.05, 0], [0, 0.05]])
    R = np.array([0.5]).reshape(1, 1)
    x = np.linspace(-10, 10, 100)
    measurements = - (x**2 + 2*x - 2) + np.random.normal(0, 5, 100)
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
    predictions = []
    for z in measurements:
        kf.predict()
        kf.correct(z)
        predictions.append(kf.x[0][0])
    plt.subplot(121)
    plt.plot(x, measurements, 'b')
    plt.plot(x, predictions, 'g')
    plt.subplot(122)
    plt.plot(x, kf.sigma00, 'b')
    plt.plot(x, kf.sigma01, 'g')
    plt.plot(x, kf.sigma11, 'r')
    plt.plot(x, kf.sigma10, 'c')
    plt.show()
