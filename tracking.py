#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detection Module.
Including point cloud processing functions, pattern extraction functions, and
others.

Notes:
    Here are some naming conventions in this code to keep simplicity.
    x_: a priori prediction of x.
    x: measurement of x(observation), but in KalmanFilter x's meaning is
    determined by the computing process.
"""
import numpy as np
from sklearn.cluster import DBSCAN, MeanShift
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from kalman import KalmanFilter
from utils import record_run_time


@record_run_time
def extract_patterns(image):
    """Use DBSCAN or other algorithms to extract patterns from a 2d point cloud.
    Parameters:
        image : array_like(dtype='float', shape=(n, 2))
            A 2d point cloud.

    Returns:
        labels : array(dtype='float', shape=(n, 1))
            Labels of the patterns.

    """
    clustering = DBSCAN().fit(image)
    labels = clustering.labels_
    return labels


def distance(x_, x, y_, y, var_x, var_y):
    return (x_ - x) ** 2 / var_x + (y_ - y) ** 2 / var_y


class MTT(object):
    """Multi-Target Tracking.
    Attributes:
        tracks : list of Track
            A list of Track object.
        T : float
            Sampling period.
        F, H, Q, R : np.array
            Parameters of kalman filter, all KalmanFilter objects will share
            the same F, H, Q, R.
    """

    def __init__(self):
        self.tracks = []
        self.T = 0.1
        T = self.T
        self.F = np.array([[1, T], [0, 1]], dtype='float')
        self.H = np.array([1, 0], dtype='float').reshape(1, 2)
        self.Q = 10 * np.array([[T**3 / 3, T**2 / 2], [T**2 / 2, T]])
        self.R = np.array([1]).reshape(1, 1)
        self.P = np.array([[4, 0], [0, 1]])

    @record_run_time
    def associate_track(self, x_m, y_m):
        """Associate tracks and gate using Munkres Algorithm.

        Parameters:
            x_m, y_m : list of float
                Observations (or measurements) to be associated with `tracks`

        Returns:
            row_ind, col_ind : list of indices
                Associated indices, `row_ind` for `tracks`, `col_ind` for
                `x_m` and `y_m`.

        Notes:
            `predict` must be evaluated for every tracks before called.
        """
        n = len(self.tracks)
        m = len(x_m)
        np.set_printoptions(precision=3, threshold=10000, suppress=True)
        cost_matrix = np.zeros(shape=(n, m), dtype='float')
        for i in range(n):
            for j in range(m):
                track = self.tracks[i]
                cost_matrix[i][j] = distance(track.x, x_m[j], track.y, y_m[j],
                                             track.var_x, track.var_y)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        print(cost_matrix)
        print(row_ind, col_ind)
        row_ind_ = []
        col_ind_ = []
        for i, j in zip(row_ind, col_ind):
            track = self.tracks[i]
            upper_bound = 3 * np.sqrt(sum([track.R_x, track.R_y,
                                           track.var_x, track.var_y]))
            if cost_matrix[i][j] <= upper_bound:  # perform gating
                row_ind_.append(i)
                col_ind_.append(j)
            print("p:(%+.3f, %+.3f) m:(%+.3f, %+.3f) d_kj:%+.3f ub:%+.3f" % (
                   track.x, track.y, x_m[j], y_m[j],
                   cost_matrix[i][j], upper_bound))
            print("var_x:%+.3f, var_y:%+.3f" % (track.var_x, track.var_y))

        # x = input("save?")
        # if x == 'y':
            # np.savetxt('cost_matrix.csv', cost_matrix, delimiter=',')

        return row_ind_, col_ind_

    @record_run_time
    def maintain_track(self, row_ind, col_ind):
        """Confirm or delete tracks.

        Parameters:
            row_ind, col_ind : list of indices
                Returns from `associate_track`.

        Notes:
            This function will change `self.tracks`, so all indices will be
            invalid after calling it.
        """
        print("final row_ind: %s" % row_ind)
        for i, track in enumerate(self.tracks):
            track.mark_whether_assigned(assigned=i in row_ind)

        self.tracks = list(filter(lambda t: not t.deleted, self.tracks))

    @record_run_time
    def run(self, img):
        """
        Parameters:
            img : array(shape=(n, 2))
                points in the form of (x, y)
        Returns:
            labels : array(shape=(n, 1))
                labels of every points, returned from DBSCAN.
        """
        labels = extract_patterns(img)
        n = np.max(labels)

        x_m = []  # measurements of x
        y_m = []
        for i in range(n):
            cluster = img[labels == i]
            x_m.append(np.average(cluster[:, 0]))
            y_m.append(np.average(cluster[:, 1]))

        if self.tracks == []:  # initialize kalman filters
            for i in range(n):
                x0 = np.array([x_m[i], 0]).reshape(2, 1)
                y0 = np.array([y_m[i], 0]).reshape(2, 1)
                kf_x = KalmanFilter(self.F, self.H, self.Q, self.R,
                                    P=self.P.copy(), x0=x0)
                kf_y = KalmanFilter(self.F, self.H, self.Q, self.R,
                                    P=self.P.copy(), x0=y0)
                track = Track(kf_x, kf_y)
                self.tracks.append(track)
        else:
            for track in self.tracks:
                track.predict()

            row_ind, col_ind = self.associate_track(x_m, y_m)

            for i, j in zip(row_ind, col_ind):
                self.tracks[i].correct(x_m[j], y_m[j])

            self.maintain_track(row_ind, col_ind)

            # add(initialize) the tracks if any observations(measurements) are
            # not assigned with a tracks.
            for i in set(range(len(x_m))).difference(col_ind):
                x0 = np.array([x_m[i], 0]).reshape(2, 1)
                y0 = np.array([y_m[i], 0]).reshape(2, 1)
                kf_x = KalmanFilter(self.F, self.H, self.Q, self.R,
                                    P=self.P.copy(), x0=x0)
                kf_y = KalmanFilter(self.F, self.H, self.Q, self.R,
                                    P=self.P.copy(), x0=y0)
                track = Track(kf_x, kf_y)
                self.tracks.append(track)
        return labels


class Track(object):
    """2-D Track.

    Attributes:
        kf_x, kf_y : KalmanFilter
            We use two KalmanFilter objects to represent a track.
        deleted : bool (default=False)
            Used to mark the track's status, since Track is not able to delete
            itself. Only `mark_whether_assigned` will change it.
        mn_ad_hoc_rule, m, n : list of int, int, int
            Used to implement M/N ad hoc rule.
        hists : list of Hist.
            See class Hist.
    """

    def __init__(self, kf_x, kf_y):
        self.kf_x = kf_x
        self.kf_y = kf_y
        self.deleted = False
        self.mn_ad_hoc_rule = []
        self.m = 5
        self.n = 2
        self.hists = []

    def predict(self):
        kf_x = self.kf_x
        kf_y = self.kf_y
        kf_x.predict()
        kf_y.predict()
        hist = Hist(x_=kf_x.x[0][0], vx_=kf_x.x[1][0], Px_=kf_x.P.copy(),
                    y_=kf_y.x[0][0], vy_=kf_y.x[1][0], Py_=kf_y.P.copy())
        self.hists.append(hist)

    def correct(self, x, y):
        self.kf_x.correct(x)
        self.kf_y.correct(y)
        self.hists[-1].x = x
        self.hists[-1].y = y

    def mark_whether_assigned(self, assigned):
        self.mn_ad_hoc_rule.append(0 if assigned else 1)
        if len(self.mn_ad_hoc_rule) > self.m:
            self.mn_ad_hoc_rule = self.mn_ad_hoc_rule[1:]
        self.deleted = True if sum(self.mn_ad_hoc_rule) >= self.n else False

    @property
    def x(self):
        return self.kf_x.x[0][0]

    @property
    def y(self):
        return self.kf_y.x[0][0]

    @property
    def R_x(self):
        return self.kf_x.R[0][0]

    @property
    def R_y(self):
        return self.kf_y.R[0][0]

    @property
    def var_x(self):
        return self.kf_x.P[0][0]

    @property
    def var_y(self):
        return self.kf_y.P[0][0]


class Hist(object):
    __slots__ = (
            'x_', 'vx_', 'x', 'Px_',
            'y_', 'vy_', 'y', 'Py_'
            )

    def __init__(self, x_=None, vx_=None, x=None, Px_=None,
                 y_=None, vy_=None, y=None, Py_=None):
        self.x_ = x_
        """prediction of x"""
        self.vx_ = vx_
        """prediction of velocity of x"""
        self.x = x
        """measurement of x, could be `None`. If it is the case, `correct` is
        not performed due to failing matching any obeservations."""
        self.y_ = y_
        self.vy_ = vy_
        self.y = y
        self.Px_ = Px_
        self.Py_ = Py_


class BBox(object):
    """Minimum bounding box.

    Parameters:
        mn : np.array
            Minimum value along every axis.
        mx : np.array
            Maximum value along every axis.
    """

    def __init__(self, mn=None, mx=None):
        self.mn = mn
        self.mx = mx

    def is_none(self):
        return self.mn is None or self.mx is None
