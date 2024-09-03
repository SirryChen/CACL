# CACL: Community-Aware Heterogeneous Graph Contrastive Learning for Social Media Bot Detection

>Accepted by ACL 2024 Findings

CACL is a Community-Aware Heterogeneous Graph Contrastive Learning framework and we apply it to social media bot detection.

The implementation of CACL is mainly based on [Pytorch](https://github.com/pytorch/pytorch) and [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) API.


## Overview

The steps to reimplement this work mainly contain:

- Data preprocessing  
  - Three datasets are used and programs for preprocessing can be found as `predata_{dataset name}.py`.
  - Just download the datasets and change the file paths augments.

- Pretrain  
  - Community-aware module is employed to perform community detection and link prediction, which needs to be pretrained to prevent cold-start problem.
  - Simply change the preprocessed data path augment and use `pretrain.py`.

- Train  
  - After obtaining preprocessed data and pretrained community-aware module weight, use `train.py` to train the full model.
  - Hyperparameters can be adjusted in the function `super_parament_initial()` in `utils.py`.

- Test  
After training the model using `train.py`, the test results will be shown and saved.

## Datasets

We have used three datasets throughout the entire work. You may need to contact the author to get access to some of them.

- Cresic-15
[[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0167923615001803?via%3Dihub)
[[dataset]](https://www.researchgate.net/figure/Details-of-the-Cresci-2015-dataset_tbl2_370918138)

- Twibot-20
[[paper]](https://dl.acm.org/doi/10.1145/3459637.3482019)
[[github]](https://github.com/BunsenFeng/TwiBot-20)

- Twibot-22
[[paper]](https://papers.nips.cc/paper_files/paper/2022/hash/e4fd610b1d77699a02df07ae97de992a-Abstract-Datasets_and_Benchmarks.html)
[[github]](https://github.com/LuoUndergradXJTU/TwiBot-22)
[[dataset]](https://drive.google.com/drive/folders/1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs)

- Other datasets may be helpful
[[web path]](https://botometer.osome.iu.edu/bot-repository/datasets.html)

## Training details
Using dataset Cresci-15 and backbone HGT for example.

Once you preprocess the dataset using `predata_cresci15.py`, then you can:
- Pretrain the community-aware module 
```bash
python3 pretrain.py --dataset cresci15 --basic_model HGT
```

- Train the CACL with HGT
```bash
python3 pretrain.py --dataset cresci15 --basic_model HGT --max_error_times 5
```

Here are some key options of the hyperparameters

- `basic_model`: CACL framework support 3 backbones as the convolutional layer including GAT, SAGE, and HGT. 
- `num_layer`: the layer number of the convolutional network, we use 2 by default.
- `lr_warmup_epochs`: during the initial training warm-up phase of the model, we increase the weight of the contrastive loss, aiming for the model to quickly find the optimal point.
- `max_error_times`: we use the validation dataset for early stopping.
- `cluster`: we implement several cluster method for community detection, we use randomwalk by default.

The details of other optional hyperparameters can be found in the function `super_parament_initial()` in [`utils.py`](./utils.py)

## Citation

Please consider citing the following paper when using our code for your application.

```bibtex
@inproceedings{CACL2024,
  author       = {Sirry Chen and
                  Shuo Feng and
                  Songsong Liang and
                  Chen{-}Chen Zong and
                  Jing Li and
                  Piji Li},
  editor       = {Lun{-}Wei Ku and
                  Andre Martins and
                  Vivek Srikumar},
  title        = {{CACL:} Community-Aware Heterogeneous Graph Contrastive Learning for
                  Social Media Bot Detection},
  booktitle    = {Findings of the Association for Computational Linguistics, {ACL} 2024,
                  Bangkok, Thailand and virtual meeting, August 11-16, 2024},
  pages        = {10349--10360},
  publisher    = {Association for Computational Linguistics},
  year         = {2024},
  url          = {https://aclanthology.org/2024.findings-acl.617},
  timestamp    = {Tue, 27 Aug 2024 17:38:11 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/ChenFLZLL24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

