#!/usr/bin/env python3
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-f",
            "--pcap-file",
            default="data/2019-07-15-10-23-00-RS-16-Data.pcap",
            help="read from pcap file")

    parser.add_argument(
            "-r",
            "--reset-frames",
            action="store_true",
            help="parse pcap only and serialize it, but do not analyze frames"
            )
    parser.add_argument(
            "-s",
            "--use-serialized-frame",
            action="store_true",
            help="use serialized frame instead of parsing the pcap"
            )
    parser.add_argument(
            "-m",
            "--mark-contour-data",
            action="store_true",
            help="mark contour data and serialize it"
            )
    parser.add_argument(
            "--show-frame",
            action="store_true",
            help="show 3d scatter of a frame for every loop"
            )
    args = parser.parse_args()

import dpkt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os
import pickle

import pointcloud_parser
from utils import record_run_time, recording, show_run_time, save_frame,\
        load_frame, plot3d, explore_images, my_imshow
from tracking import mtt


def filter_bound(p, xlim, ylim, zlim):
    return p[(p[:, 0] > xlim[0]) & (p[:, 0] < xlim[1]) & (p[:, 1] > ylim[0]) &
             (p[:, 1] < ylim[1]) & (p[:, 2] > zlim[0]) & (p[:, 2] < zlim[1])]


@record_run_time
def grid_image(pointcloud, resolution, mn=None, mx=None):
    """Convert the point clouds into a image.

    Parameters:
        pointcloud : np.array(shape=(n, 4))
        resolution : float
            The unit is meter.
        mn, mx : np.array(shape=(2,))
            The bound of point cloud.

    Returns:
        (np.array(dtype='uint8'), np.array, np.array): the first is the
        gridded image; the second and the third are respectively x/y-axis
        gridded coordinates."""
    xy = pointcloud.T[:2]
    xy = ((xy + resolution / 2) // resolution).astype(int)
    if mn is None or mx is None:
        mn, mx = xy.min(axis=1), xy.max(axis=1)
    sz = mx + 1 - mn
    flatidx = np.ravel_multi_index(xy - mn[:, None], sz)
    with recording("np.bincount"):
        histo = np.bincount(flatidx, pointcloud[:, 3], sz.prod()) / np.maximum(
            1, np.bincount(flatidx, None, sz.prod()))
    return (histo.reshape(sz).astype('uint8'), *xy)


@record_run_time
def analyze_frame(frame, args):

    # filter frame
    frame = filter_bound(frame, [-10., 6.], [-4., 5.], [-3., 3])

    # discretize the frame
    res_xy = 0.02
    res_z = 10

    def discretize(x, res):
        return ((x + res / 2) // res).astype(int)

    if args.show_frame:
        print("show_frame: number of points in frame is %s" % frame.shape[0])
        plot3d(frame)
        plt.show()

    # get mn, mx to fix region of a image
    xy = frame.T[:2]
    xy = discretize(xy, res_xy)
    mn, mx = xy.min(axis=1), xy.max(axis=1)
    images = slice_vertical(frame, res_z)

    # analyze image; get analysis results and show
    lst = [
        analyze_image(image, res_xy, mn, mx) + [image] for image in images
        if len(image) > 0
    ]
    imgrays, imcnts, contours_lst, images = tuple(map(list, zip(*lst)))

    if args.mark_contour_data:
        f_name = "./serial/contours.p"
        if os.path.exists(f_name):
            with open(f_name, 'rb') as f:
                cnt_saved = pickle.load(f)
        else:
            cnt_saved = []

        reg = explore_images((imgrays, imcnts, images), (0, 0, 1), 2, 2, mn,
                             mx, res_xy)
        reg_d = discretize(reg, res_xy) - np.tile(mn, 2)
        assert np.all(reg_d >= 0)
        for imcnt, contours in zip(imcnts, contours_lst):
            imcnt_part = imcnt[reg_d[0]:reg_d[2], reg_d[1]:reg_d[3]]
            if len(imcnt_part) > 0:
                fig = plt.figure()
                ax = fig.add_subplot(111, aspect='equal')

                # analyze contour
                def filter_contour(cnt):
                    # opencv treat arrays of image as [y, x, color],
                    # and we defines pointclouds like [x, y, z, intensity],
                    # so we need to exchange x and y
                    y, x, h, w = cv2.boundingRect(cnt)
                    x -= reg_d[0]
                    y -= reg_d[1]
                    return (w * h > 8 and any([
                        reg_d[0] < p[0][1] < reg_d[2]
                        and reg_d[1] < p[0][0] < reg_d[3] for p in cnt
                    ]))

                contours_filtered = [
                    cnt for cnt in contours if filter_contour(cnt)
                ]

                for i, cnt in enumerate(contours_filtered):
                    y, x, h, w = cv2.boundingRect(cnt)
                    x -= reg_d[0]
                    y -= reg_d[1]
                    ax.add_patch(
                        patches.Rectangle(
                            (x - 0.5, y - 0.5),
                            w,
                            h,
                            edgecolor='r',
                            fill=None
                            ))
                    plt.text(x - 0.5, y + h - 0.5, f"{i}", color='b')

                c = contours_filtered
                k = 230
                for i in range(len(c)):
                    print(f"distance({k}, {i}): "
                          f"{cv2.matchShapes(c[k], c[i], 3, 0.)}")

                my_imshow(imcnt_part)
                plt.show()
                i = int(input("input the index of contours you want to save:"))
                cnt_saved += [contours_filtered[i]]
                with open(f_name, 'wb') as f:
                    pickle.dump(cnt_saved, f)

    # x =  img[:,0]; y = img[:,1]; z = img[:,2]
    # pitch = img[:,3]  # reflexity of laser
    # plt.subplot(1, 1, 1)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.plot(x, y, 'go', markersize=2)

    # model = MA(10,4)
    # order = np.argsort(x)
    # plt.plot(x[order], model.conv1d(y[order]), 'ro', markersize=1)

    # if False:
    # model = PolynomialFit(2)
    # loss = model.train(x, y, z)
    # print('loss: %s' % loss)

    # plt.xlim([-40, 40])
    # plt.ylim([-40, 40])
    # ax.scatter(x, y, z)

    # x_ = np.arange(-40, 40, 4)
    # y_ = np.arange(-40, 40, 4)
    # x_, y_ = np.meshgrid(x_, y_)
    # z_ = model.predict(x_, y_)
    # ax.plot_surface(x_, y_, z_, cmap=cm.coolwarm)


@record_run_time
def slice_vertical(frame, resolution):
    """Slice a 3d frame into 2d images vertically.

    Parameters:
        frame : np.array(shape=(n, 4))
            All columns are respectively x, y, x, intensity.
        resolution : float
            The unit is meter.

    Returns:
        [np.array]: list of np.array of shape (k, 4), all k should add up to n.
        Every element is a slice of frame, and are bounded by (z_max, z_min).
    """
    z = frame[:, 2]
    z = ((z + resolution / 2) // resolution).astype(int)
    mn, mx = z.min(), z.max()
    sz = mx + 1 - mn
    return [frame[z - mn == i] for i in range(sz)]


def gray2rgb(im):
    """cv2 aided function"""
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)


def gray2bgr(im):
    """cv2 aided function"""
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def bgr2rgb(im):
    """cv2 aided function"""
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


@record_run_time
def analyze_image(img, resolution, mn=None, mx=None):
    mtt.run(img[:, [0, 1]])
    imgray, x, y = grid_image(img, resolution, mn, mx)
    # print(imgray.shape)
    # cv2.imshow('greyimage', imgray)
    # cv2.waitKey()
    ret, thresh = cv2.threshold(imgray, 1, 255, cv2.THRESH_BINARY)
    imcnt = gray2bgr(thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    contours = np.array(contours)

    cv2.drawContours(imcnt, contours, -1, (0, 255, 0), 1)

    return [imgray, bgr2rgb(imcnt), contours]


def test(args):
    if args.use_serialized_frame:
        for frame in load_frame():
            analyze_frame(frame, args)
            if args.mark_contour_data:
                r = input('Continue?')
            else:
                r = 'y'
            if r == 'y':
                continue
            else:
                break
        return

    with open(args.pcap_file, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        i = 0
        for timestamp, buf in pcap:
            if len(buf) == 1248:
                frame = pointcloud_parser.parse(buf, 0)
            elif len(buf) == 1290:
                frame = pointcloud_parser.parse(buf[42:], 1)

            if frame is None:
                continue

            if args.reset_frames:
                save_frame(frame.astype('f4'), i)
                i += 1
                continue
            analyze_frame(frame, args)
            if args.mark_contour_data:
                r = input('Continue?')
            else:
                r = 'y'
            if r == 'y':
                i += 1
                continue
            else:
                i += 1
                break


if __name__ == "__main__":
    test(args)
    show_run_time()
