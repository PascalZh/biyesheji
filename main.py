#!/usr/bin/env python3
import parser
from utils import *

class PolynomialFit():

    def __init__(self, n):
        self.c = np.random.randn((n+2)*(n+1)//2).reshape(-1,1)
        self.n = n
        print(self.c.shape)

    def train(self, x, y, z):
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        z = z.reshape(-1,1)
        for i in range(self.n+1):
            if i == 0:
                X = np.ones(x.shape)
                continue
            for j in range(i+1):
                X = np.concatenate((X, x**j*(y**(i-j))), axis=1)
        self.c = matmul(matmul(linalg.inv(matmul(X.transpose(), X)), \
                X.transpose()), z)
        d = matmul(X, self.c) - y
        loss = matmul(d.transpose(), d) / x.shape[0]
        return loss.reshape(1)
    def predict(self, x, y):
        shape = x.shape
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        for i in range(self.n+1):
            if i == 0:
                X = np.ones(x.shape)
                continue
            for j in range(i+1):
                X = np.concatenate((X, x**j*(y**(i-j))), axis=1)
        return matmul(X, self.c).reshape(shape)


class MA():
    def __init__(self, wnd_size):
        self.kernel = np.ones(wnd_size) / wnd_size
    def conv1d(self, x):
        return np.convolve(x.reshape(-1), self.kernel, 'same')


@record_run_time
def process_image(img, test=False):
    x =  img[:,0]; y = img[:,1]; z = img[:,2]
    pitch = img[:,3]  # reflexity of laser
    #plt.subplot(2, 1, 1)
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.plot(x, y, 'ro')

    #model = MA(5)
    #plt.plot(x, model.conv1d(y), 'go')

    if False:
        model = PolynomialFit(2)
        loss = model.train(x, y, z)
        print('loss: %s' % loss)

        plt.xlim([-40, 40])
        plt.ylim([-40, 40])
        ax.scatter(x, y, z)

        x_ = np.arange(-40, 40, 4)
        y_ = np.arange(-40, 40, 4)
        x_, y_ = np.meshgrid(x_, y_)
        z_ = model.predict(x_, y_)
        ax.plot_surface(x_, y_, z_, cmap=cm.coolwarm)

        plt.show()


def split2d(f, z_min, z_max, num):
    f = f[f[:,2].argsort()]
    z = f[:,2]
    t = (z_max - z_min) / num
    s = np.searchsorted
    return [f[s(z, z_min+i*t):s(z, z_min+(i+1)*t)] for i in range(num)]

def test():
    """Open vlp pcap file and print out the packets"""
    #with open('data/2015-07-23-14-37-22_Velodyne-VLP-16-Data_Downtown 10Hz Single.pcap', 'rb') as f:
    with open('data/2019-07-15-10-23-00-RS-16-Data.pcap', 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        i = 0
        for timestamp, buf in pcap:
            if len(buf) == 1248:
                frame = parser.parse(buf, 0)
            elif len(buf) == 1290:
                frame = parser.parse(buf[42:], 1)

            if frame is None:
                continue
            else:
                images = split2d(frame, -6, 6, 100)
                image = images[30]
                i += 1
                if i >= 10000:
                    break

    #mlab.points3d(frame[:,0], frame[:,1], frame[:,2], frame[:,3], mode='point', colormap='cool')
    #mlab.show()
    process_image(image, test=True)
    show_run_time()

if __name__ == '__main__':
    test()
