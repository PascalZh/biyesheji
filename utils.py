import dpkt
import time
from numba import jit
import numpy as np
from numpy import linalg, matmul, sin, cos, pi

from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

run_time = dict()


def print_hex(bytes):
    l = [hex(int(i)) for i in bytes]
    print(" ".join(l[:84]))
    for i in range(12):
        print(" ".join(l[84+i*100:84+i*100+100]))

def record_run_time(func):
    def call_func(*args, **kwargs):
        global run_time
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
    global run_time
    print('================SHOW RUN TIME=================')
    for f in run_time.keys():
        t = run_time[f][0]; n = run_time[f][1]
        info = "function \033[36m%s\033[0m \truns for \033[34m%.3fms\033[0m, \
\033[32m%.5fμs\033[0m a time, %s times."  % (f, t*1000, t*1e6/n, n)
        with open('run_time.log', 'a') as f:
            f.write(info+"\n")
        print(info)
    with open('run_time.log', 'a') as f:
        f.write("\n")

@record_run_time
def append_0(a, b):
    if a is None:
        return b
    else:
        return np.append(a, b, axis=0)

'''
    z_min = -10; z_max = 10
    images = split2d(frames[0], z_min, z_max, 10)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    l, = plt.plot(images[0,:,0], images[0,:,1], ',g')
    class Callback(object):
        def __init__(self, text):
            self.text = text
            self.i = 0
            text.set_val(self.i)
        def next(self, event):
            self.i += 1
            if self.i not in range(len(images)):
                self.i = len(images)-1
            i = self.i
            self.text.set_val('%s' % i)
            l.set_xdata(images[i,:,0])
            l.set_ydata(images[i,:,1])
            plt.draw()
        def prev(self, event):
            self.i -= 1
            if self.i not in range(len(images)):
                self.i = 0
            i = self.i
            self.text.set_val('%s' % i)
            l.set_xdata(images[i,:,0])
            l.set_ydata(images[i,:,1])
            plt.draw()
    axtext = plt.axes([0.2, 0.05, 0.1, 0.075])
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    text = TextBox(axtext, 'number:')
    callback = Callback(text)
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()
'''
