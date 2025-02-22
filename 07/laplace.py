#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    x = np.linspace(-4,8,90)
    fx = -0.5*(x-2)**2+1
    qx = np.exp(-0.5*(x-2)**2)
    plt.plot(x, fx)
    plt.plot(x, qx)
    plt.ylim(0,1)
    plt.show()
    
    
