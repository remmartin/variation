import sys
import math
import copy
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal


def rgb2gray(rgb):
    if (len(rgb.shape) == 3):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = rgb
    return gray

def gkern(kernlen=11, nsig=2):
    lim = kernlen//2 + (kernlen % 2)/2
    x = np.linspace(-lim, lim, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class Minimisation:
    def __init__(self, image, kern, n):
        self._init_image = copy.deepcopy(image)
        self._current_image = copy.deepcopy(image)
        self._init_kernel = copy.deepcopy(kern)
        self._noise = n

        self._discrepancy = 0
        self._tgv = 0

        self._mu = 0.85
        self._v = 0
        return

    def _count_grad(self):
        R = [[1, 0], [0, 1], [1, 1], [1, -1]]
        tgv = 0
        if (self._noise == 0):
            alpha1 = 0.0003
            alpha2 = 0.00005
        elif (self._noise <= 3):
            alpha1 = 0.0015
            alpha2 = 0.0007
        elif (self._noise < 10):
            alpha1 = (0.2 * self._noise - 1) / 100
            alpha2 = (0.2 * self._noise - 1) / 100
        elif (self._noise < 15):
            alpha1 = 0.0015
            alpha2 = 0.0015
        else:
            alpha1 = 0.05
            alpha2 = 0.05


        for cur_R in R:
            btv = np.sign(np.roll(self._current_image + self._mu * self._v, (cur_R[0], cur_R[1]), axis=(1, 0))
                            - self._current_image)
            btv2 = np.sign(np.roll(self._current_image + self._mu * self._v, (cur_R[0], cur_R[1]), axis=(1, 0)) +
                            np.roll(self._current_image + self._mu * self._v, (-cur_R[0], -cur_R[1]), axis=(1, 0))
                            - 2 * self._current_image)
            tgv += (1 / math.sqrt(cur_R[0] ** 2 + cur_R[1] ** 2)) \
                      * (alpha1 * (np.roll(btv, (- cur_R[0], - cur_R[1]), axis=(1, 0)) - btv)
                      + alpha2 * (np.roll(btv2, (- cur_R[0], - cur_R[1]), axis=(1, 0))
                                  + np.roll(btv2, (cur_R[0], cur_R[1]), axis=(1, 0)) - 2 * btv2))

        self._tgv = tgv
        tmp = signal.convolve2d(self._current_image + self._mu * self._v, self._init_kernel, mode='same', boundary='symm') \
              - self._init_image
        self._discrepancy = 2 * signal.convolve2d(tmp, self._init_kernel, mode='same', boundary='symm')

        return

    def _nesterov(self, beta=1):
        '''
        Nesterov II
        g_k = grad(f(z_k + mu * v_k))
        v_(k+1) = mu * v_k - beta_k * g_k
        z_(k+1) = z_k + v_(k+1)
        '''
        g = (self._discrepancy + self._tgv)
        self._v = self._mu * self._v - beta * (self._discrepancy + self._tgv)

    def _update_image(self):
        self._current_image += self._v

    def Process(self):
        for i in range(1, 100):
            self._count_grad()
            #beta = 0.05 / math.sqrt(i)
            g = (self._discrepancy + self._tgv)
            if (self._noise < 3):
                beta = 0.1 * (0.5 ** (i/100))/np.sum(g**2)**0.5
            else:
                beta = 0.1 * (0.2 / 15 * noise + 0.1 / 3) * (0.5 ** (i / 100)) / np.sum(g ** 2) ** 0.5
            self._nesterov(beta)
            self._update_image()
            #print("i =", i)
            #print("beta =", beta)
            #print()


if __name__ == "__main__":
    kernel = mpimg.imread(sys.argv[2])
    original_image = mpimg.imread(sys.argv[1])
    temp_kernel = rgb2gray(kernel)
    temp_kernel /= temp_kernel.sum()
    temp_image = rgb2gray(original_image)

    noise = float(sys.argv[4])

    if (noise > 3):
        gauss_kernel = gkern(11, 2)
        image_1 = signal.convolve2d(temp_image, gauss_kernel, mode='same', boundary='symm')
        r = Minimisation(image_1, temp_kernel, noise)
        r.Process()
        q = Minimisation(r._current_image, gauss_kernel, 0)
        q.Process()
    else:
        q = Minimisation(temp_image, temp_kernel, noise)
        q.Process()

    plt.imsave(sys.argv[3], q._current_image, cmap='gray')
