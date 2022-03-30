import os

if __name__ == '__main__':
    net = 'Re_CNN_mf'
    dims = [1, 30, 50, 70, 100, 150, 200, 250, 300, 400, 512]
    seeds = [0, 1, 2, 3, 4, 5]
    kls = [0.1, 0.5, 1., 5., 10., 20.]
    for seed in seeds:
        for dim in dims:
            for kl in kls:
                # else:
                print(
                    'python train.py -net {} -dims {} -lr {} -epoch {} -if_ {} -kl {} -seed {}'.format(net, dim, 0.0005, 120, 'hipp',
                                                                                              kl, seed))
                os.system(
                    'python train.py -net {} -dims {} -lr {} -epoch {} -if_ {} -kl {} -seed {}'.format(net, dim, 0.0005, 120, 'hipp',
                                                                                              kl, seed))
