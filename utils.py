import time
import os
from contextlib import contextmanager
from functools import reduce

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox, RectangleSelector
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab

def enclosure_run_time():
    run_time = dict()

    def record_run_time(func):
        def call_func(*args, **kwargs):
            nonlocal run_time
            start = time.time()
            ret = func(*args, **kwargs)
            span = time.time() - start
            name = func.__name__
            if name not in run_time.keys():
                run_time[name] = [span, 1]
            else:
                run_time[name][0] += span
                run_time[name][1] += 1
            return ret
        return call_func

    def show_run_time():
        nonlocal run_time
        print("================SHOW RUN TIME=================")
        for f in run_time.keys():
            t, n = run_time[f][0], run_time[f][1]
            info = "function \033[36m%s\033[0m \truns for \033[34m%.3fms\033[0m, \
    \033[32m%.5fμs\033[0m a time, %s times." % (f, t*1000, t*1e6/n, n)
            print(info)

    @contextmanager
    def recording(_str):
        nonlocal run_time
        name = 'code block: "%s"' % _str
        start = time.time()
        yield
        span = time.time() - start
        if name not in run_time.keys():
            run_time[name] = [span, 1]
        else:
            run_time[name][0] += span
            run_time[name][1] += 1

    return record_run_time, recording, show_run_time


record_run_time, recording, show_run_time = enclosure_run_time()


def load_frame():
    filename = "./serial/frames"
    if not os.path.exists(filename):
        raise
    files = []
    for root, dirs, f in os.walk(filename):
        files += [f]
    if len(files) != 1:
        raise
    n = len(files[0])

    def generator(n):
        i = 0
        while True:
            if i < n:
                yield np.load(filename + "/%s.npy" % i)
                i += 1
            else:
                break
    return generator(n)


def save_frame(x, i):
    filename = "./serial/frames"
    if not os.path.exists(filename):
        os.makedirs("./serial/frames")
    np.save(filename + "/%s" % i, x)


def save_folders(folder, subfolder, x):
    """save x in folder/subfolder automatically named"""
    filename = folder + '/' + subfolder
    if not os.path.exists(filename):
        os.makedirs(filename)
    files = list(os.walk(filename))[0][2]
    n = -1
    if len(files) != 0:
        n = reduce(max, [int(file_.split('.')[0]) for file_ in files])
    np.save(filename + '/%05d.npy' % (n + 1), x)


def load_folders(folder, subfolder):
    filename = folder + '/' + subfolder
    files = list(os.walk(filename))[0][2]
    return [np.load(filename + '/' + f) for f in files]


def print_hex(bytes_):
    lst = [hex(int(i)) for i in bytes_]
    print(" ".join(lst[:84]))
    for i in range(12):
        print(" ".join(lst[84+i*100:84+i*100+100]))


@record_run_time
def append_0(a, b):
    if a is None:
        return b
    else:
        return np.append(a, b, axis=0)


def my_imshow(im):
    plt.imshow(im.transpose(1, 0, 2), origin='lower')


def plot3d(f):
    fig = plt.figure("point clouds view")
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    ax.scatter(f[:, 0], f[:, 1], f[:, 2], s=2, c=f[:, 3])
    fig.show()
    # mlab.points3d(f[:,0], f[:,1], f[:,2], f[:,3], mode='point', colormap='cool')
    # mlab.axes(xlabel='x', ylabel='y', zlabel='z')
    # mlab.show()


def explore_images(images_tuple, draw_type, rows, cols, mn, mx, res):
    extent = res * np.array([mn[0], mx[0], mn[1], mx[1]])
    print(extent)
    xlim = ((mn[0] - 1) * res,  (mx[0] + 1) * res)
    ylim = ((mn[1] - 1) * res,  (mx[1] + 1) * res)
    n = len(images_tuple)
    n_i = len(images_tuple[0])

    fig = plt.figure("projected images")
    fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.1, top=0.99, right=0.95,
                        wspace=0.01, hspace=0.01)

    axes_image = []
    axes = []

    for j in range(n):
        if j > 0:
            sharex, sharey = axes[0], axes[0]
        else:
            sharex, sharey = None, None
        ax = fig.add_subplot(rows, cols, j+1, sharex=sharex, sharey=sharey)
        axes.append(ax)
        plt.xlabel("x/m", fontsize=9)
        plt.ylabel("y/m", fontsize=9)
        plt.xlim(xlim)
        plt.ylim(ylim)
        ax.label_outer()
        if draw_type[j] == 0:
            img = images_tuple[j][0]
            if len(img.shape) == 3:
                img = img.transpose(1, 0, 2)
            elif len(img.shape) == 2:
                img = img.transpose(1, 0)
            a = plt.imshow(img, extent=extent, cmap='gray', origin='lower')
            axes_image += [a]
        elif draw_type[j] == 1:
            x = images_tuple[j][0][:, 0]
            y = images_tuple[j][0][:, 1]
            ax.set_aspect((ylim[1]-ylim[0]) / (xlim[1]-xlim[0]) /
                          ax.get_data_ratio())
            plt.plot(x, y, 'b,')
            axes_image += [None]

    class Index():

        def __init__(self, text):
            self.text = text
            self.i = 0
            text.set_val(self.i)

        def refresh_window(self):
            i = self.i
            self.text.set_val('%s' % i)
            for j in range(n):
                if draw_type[j] == 0:
                    img = images_tuple[j][i]
                    if len(img.shape) == 3:
                        img = img.transpose(1, 0, 2)
                    elif len(img.shape) == 2:
                        img = img.transpose(1, 0)
                    axes_image[j].set_data(img)
                    plt.draw()
                elif draw_type[j] == 1:
                    x = images_tuple[j][i][:, 0]
                    y = images_tuple[j][i][:, 1]
                    plt.subplot(axes[j])
                    plt.cla()
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    plt.plot(x, y, 'b,')

        def next(self, event):
            self.i += 1
            if self.i not in range(n_i):
                self.i = n_i - 1
            self.refresh_window()

        def prev(self, event):
            self.i -= 1
            if self.i not in range(n_i):
                self.i = 0
            self.refresh_window()

    axtext = plt.axes([0.2, 0.005, 0.1, 0.035])
    axprev = plt.axes([0.7, 0.005, 0.1, 0.035])
    axnext = plt.axes([0.81, 0.005, 0.1, 0.035])
    text = TextBox(axtext, 'number:')
    callback = Index(text)
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    # explain: region_selected = [xmin, ymin, xmax, ymax]
    region_selected = np.array([.0, .0, .0, .0])

    def onselect(eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        print('left bottom position: (%f, %f)' %
              (eclick.xdata, eclick.ydata))
        print('right bottom position: (%f, %f)' %
              (erelease.xdata, erelease.ydata))
        print('used button  : ', eclick.button)
        region_selected[0], region_selected[1] = eclick.xdata, eclick.ydata
        region_selected[2], region_selected[3] = erelease.xdata, erelease.ydata
        print('region_selected saved: %s' % region_selected)

    def toggle_selector(event):
        print('Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print('RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print('RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

    toggle_selector.RS = RectangleSelector(axes[2], onselect)
    fig.canvas.mpl_connect('key_press_event', toggle_selector)

    # plt.show() must be here, or the controls doesn't work.
    # To show other figures at the same time, call plt.show() more than
    # once will cause the figures doesn't show at the same time
    # The only way to resolve it is to write fig.show() in other place,
    # remaining only one plt.show() here.
    plt.show()
    return region_selected
