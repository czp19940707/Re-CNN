# Re_CNN_master
A Reparametrized CNN Model to Distinguish Alzheimer's Disease Applying Multiple Morphological Metrics and Deep Semantic Features from Structural MRI.
![image](https://github.com/czp19940707/Re_CNN_master/blob/main/Re-CNN.png)
# DATASET
The data_path should look like:

    data
    ├── ADNI
    │   ├── images
    │   │       ├── AD
    │   │       │    ├── imageid.npy
    │   │       │    └── ...
    │   │       └── CN
    │   │            ├── imageid.npy
    │   │            └── ...
    │   ├── Morphological_Metrics
    │   │       ├── AD
    │   │       │    ├── imageid.npy
    │   │       │    └── ...
    │   │       └── CN
    │   │            ├── imageid.npy
    │   │            └── ...
    ├── AIBL
    │   ├── images
    │   │       ├── AD
    │   │       │    ├── imageid.npy
    │   │       │    └── ...
    │   │       └── CN
    │   │            ├── imageid.npy
    │   │            └── ...
    │   ├── Morphological_Metrics
    │   │       ├── AD
    │   │       │    ├── imageid.npy
    │   │       │    └── ...
    │   │       └── CN
    │   │            ├── imageid.npy
    │   │            └── ...  
    ├── seed_0
    │
    ...
    │
    │
    └── seed_5

        
# Use:

1. data download:
    ADNI: http://adni.loni.usc.edu/
    AIBL: https://aibl.csiro.au/
2. Preprocessing according to https://github.com/vkola-lab/brain2020, step 1-4
3. Unzip outputs.zip, sample.zip and Morphological.zip to the root directory
4. Morphological Metrics stored in Morphological Metrics/,
   run mat2npy.py to convert .mat into .npy.
5. run train_sys.py to train all models.
6. run test_sys.py to test all models.
7. The files in the script folder can generate all the figures and tables in manuscript.
