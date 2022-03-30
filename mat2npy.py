import os.path
import mat4py
import numpy as np

if __name__ == '__main__':
    path = r'Morphological_Metrics'
    set_ = ['ADNI', 'AIBL']
    for st in set_:
        pa_ = os.path.join(path, st)
        for cls in os.listdir(pa_):
            p_ = os.path.join(pa_, cls)
            save_path_ = os.path.join('data', st, 'Morphological_Metrics', cls[:2])
            if not os.path.exists(save_path_):
                os.makedirs(save_path_)
            for i in os.listdir(p_):
                if i.endswith('.mat'):
                    name_ = os.path.join(p_, i)
                    data_ = mat4py.loadmat(name_)['S']
                    features = np.array([])
                    for j in data_.keys():
                        if j == 'hammers':
                            for k in data_[j]['data'].keys():
                                da_ = data_[j]['data'][k]
                                da_ = np.array(da_).squeeze()
                                features = np.concatenate([features, da_])
                    np.save(os.path.join(save_path_, i.split('_')[-1][:-4] + '.npy'), features)
