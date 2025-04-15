## BrainParc: Unified Lifespan Brain Region Parcellation


> "[BrainParc: Unified Lifespan Brain Parcellation with Anatomy-guided Progressive Transmission](https://www.researchgate.net/publication/389177856_BrainParc_Unified_Lifespan_Brain_Parcellation_with_Anatomy-guided_Progressive_Transmission)", _ISMRM_, 2025, [Jiameng Liu, Feihong Liu, Kaicong Sun, Zhiming Cui, Tianyang Sun, Zehong Cao, Jiawei Huang, Shuwei Bai, Yulin Wang, Kaicheng Zhang, Caiwen Jiang, Yuyan Ge, Han Zhang, Feng Shi and <u>Dinggang Shen</u>]

> "BrainParc: Unified Lifespan Brain Parcellation from Structural MR Images", [Jiameng Liu, Feihong Liu, Kaicong Sun, Zhiming Cui, Tianyang Sun, Zehong Cao, Jiawei Huang, Shuwei Bai, Yulin Wang, Kaicheng Zhang, Caiwen Jiang, Yuyan Ge, Han Zhang, Feng Shi and <u>Dinggang Shen</u>] (Under Review)

***
<div style="text-align: center">
  <img src="figure/framework.png" width="70%" alt="BrainParc Framework">
</div>

In this work, we proposed a full-stack, precise, longitudinally-consistent framework, BrainParc, for unified lifespan brain parcellation. Please find the following steps for implementation of this work.

### Step 1: Data Preparation
* Organize the data in the following format
    ```shell
  Expriment
  ├── csvfile
  │   └── file_list.csv
  ├── data
  │   ├── HCPA
  │   │   └── sub0001
  │   │       ├── brain_sober.nii.gz
  │   │       ├── brain.nii.gz
  │   │       ├── dk-struct.nii.gz
  │   │       ├── persudo_brain.nii.gz
  │   │       ├── skull-strip.nii.gz
  │   │       ├── T1w.nii.gz
  │   │       └── tissue.nii.gz
  │   ├── HCPD
  │   └── HCPY
  └── Results
      ├── BET
      └── BrainParc
          ├── checkpoints
          ├── log
          └── pred
  ```

### Step 2: Model Training
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