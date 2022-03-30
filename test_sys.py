import os
import sys
if __name__ == '__main__':
    net = 'Re_CNN_mf'
    dims = [1, 30, 50, 70, 100, 150, 200, 250, 300, 400, 512]
    seeds = [0, 1, 2, 3, 4, 5]
    kls = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
    for seed in seeds:
        for dim in dims:
            for kl in kls:
                print('python test.py -dims {} -kl {} -net {} -seed {}'.format(dim, kl, net, seed))
                os.system('python test.py -dims {} -kl {} -net {} -seed {}'.format(dim, kl, net, seed))