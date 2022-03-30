if __name__ == '__main__':
    seed = 2
    list_ = ['test_{}.txt'.format(seed), 'train_{}.txt'.format(seed), 'valid_{}.txt'.format(seed)]
    for i in list_:
        list_new = []
        f = open(i, 'r')
        for j in f:
            list_new.append(j.strip('\n').replace('Morphological Metrics', 'Morphological_Metrics'))
        f1 = open(i, 'w')
        for k in list_new:
            f1.write(k + '\n')