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

    def __call__(self, x, y):
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

    def __init__(self, wnd_size, layer_size):
        self.kernel = np.ones(wnd_size) / wnd_size
        self.layer_size = layer_size

    @record_run_time
    def __call__(self, x):
        ret = x.reshape(-1).copy()
        for i in range(self.layer_size):
            ret = np.convolve(ret, self.kernel, 'same')
        return ret
