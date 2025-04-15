## BrainParc: Unified Lifespan Brain Region Parcellation


> "[BrainParc: Unified Lifespan Brain Parcellation with Anatomy-guided Progressive Transmission](https://www.researchgate.net/publication/389177856_BrainParc_Unified_Lifespan_Brain_Parcellation_with_Anatomy-guided_Progressive_Transmission)", _ISMRM_, 2025, [Jiameng Liu, Feihong Liu, Kaicong Sun, Caiwen Jiang, Yulin Wang, Tianyang Sun, Feng Shi and <u>Dinggang Shen</u>]

> "BrainParc: Unified Lifespan Brain Parcellation from Structural MR Images", [Jiameng Liu, Feihong Liu, Kaicong Sun, Zhiming Cui, Tianyang Sun, Zehong Cao, Jiawei Huang, Shuwei Bai, Yulin Wang, Kaicheng Zhang, Caiwen Jiang, Yuyan Ge, Han Zhang, Feng Shi and <u>Dinggang Shen</u>] (Under Review)

***
<div style="text-align: center">
  <img src="figure/framework.png" width="100%" alt="BrainParc Framework">
</div>

In this work, we proposed a full-stack, precise, longitudinally-consistent framework, BrainParc, for unified lifespan brain parcellation. Please find the following steps for implementation of this work.

### Data Preparation
* Organize the data in the following format
    ```shell
  Expriment # root folder
  ├── csvfile # folder to save training list
  │   └── file_list.csv # file list with following format [IDs, folder, fold], IDs: data name; folder: data center; fold: five fold cross-validation.
  ├── data # folder to save training data
  │   ├── HCPA # data center folder
  │   │   └── sub0001 # data folder
  │   │       ├── brain_sober.nii.gz # Sober egde maps
  │   │       ├── brain.nii.gz # Intensity image with skull-stripping
  │   │       ├── dk-struct.nii.gz # dk-struct 
  │   │       ├── persudo_brain.nii.gz # Pseudo brain for brain extraction
  │   │       ├── skull-strip.nii.gz # Brain mask
  │   │       ├── T1w.nii.gz # Intensity image without skull-stripping
  │   │       └── tissue.nii.gz # Tissue maps
  │   ├── HCPD # data center folder
  │   └── HCPY # data center folder
  └── Results # folder to save checkpoint and log file
      ├── BET # folder to save skull-stripping results and checkpoints
      └── BrainParc # folder to save brain parcellation results and checkpoints
          ├── checkpoints # folder to save checkpoint
          ├── log # folder to save logging information
          └── pred # folder to save results of validation set
  ```

### Step 1: Brain Extraction
* For model training and brain extraction inference please refer to the [AutoStrip](https://github.com/SaberPRC/AutoStrip) repo. 

### Step 2: Model Training (BrainParc)
* Please find the implementation of BrainParc in ./Code folder with following format
  ```shell
  Code
  ├── config
  │   ├── __init__.py
  │   └── config.py # Configuration for model training
  ├── dataset
  │   ├── __init__.py
  │   ├── basic.py # Basic function for data loading
  │   └── dataset.py # Data loader
  ├── network
  │   ├── __init__.py
  │   ├── basic.py # Basic function for model construction
  │   └── Joint_Parc_96.py # Main framework for progressive segmentation model
  ├── trainSegNetMS_Joint_DDP.py # Training script using DDP
  └── utils
      ├── __init__.py
      ├── loss.py # Volume-aware adaptive weight loss function
      └── utils.py
  ```

* Model training
  ```python
  python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port 10086 ./Code/trainSegNetMS_Joint_96_DDP.py --platform bme --save_path ParcJoint --file_list file_list.csv --batch_size 2 --resume -1 --weight 1
  ```
## [<font color=#F8B48F size=3>License</font> ](./LICENSE)
```shell
Copyright IDEA Lab, School of Biomedical Engineering, ShanghaiTech University, Shanghai, China.

Licensed under the the GPL (General Public License);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Repo for BrainParc: Unified Lifespan Brain Parcellation from Structural MR Images
Contact: JiamengLiu.PRC@gmail.com
```

p