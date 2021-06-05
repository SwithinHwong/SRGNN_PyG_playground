# SRGNN_PyG_Playground

A reimplementation of SRGNN. 

![image](https://github.com/SwithinHwong/SRGNN_PyG_playground/blob/master/Results.PNG)

Original code from [here](https://github.com/CRIPAC-DIG/SR-GNN). Original [paper](https://arxiv.org/abs/1811.00855).

Borrow the data preprocessing from original repo, including `diginetica` and `yoochoose`.

Using PyTorch 1.8.0, [PyTorch-Geometric 1.7](https://github.com/rusty1s/pytorch_geometric) and [tqdm](https://github.com/tqdm/tqdm).

## Data preparation

1. Download datasets used in the paper: [YOOCHOOSE](http://2015.recsyschallenge.com/challenge.html) and [DIGINETICA](http://cikm2016.cs.iupui.edu/cikm-cup). Put the two specific files named `train-item-views.csv` and `yoochoose-clicks.dat` into the folder `datasets/`

2. Change to `datasets` fold and run `preprocess.py` script to preprocess datasets. Two directories named after dataset should be generated under `datasets/`.
```bash
python preprocess.py --dataset diginetica
python preprocess.py --dataset yoochoose
```


## Training and testing
```bash
cd src
# python main.py --dataset=diginetica
nohup python -u main.py --dataset=diginetica > ../log/train_diginetica.log 2>&1 &
nohup python -u main.py --dataset=yoochoose1_4 > ../log/train_yoochoose1_4.log 2>&1 &
nohup python -u main.py --dataset=yoochoose1_64 > ../log/train_yoochoose1_64.log 2>&1 &
```

```bash
tensorboard --logdir=../log
```

## Citation

If you make advantage of the SR-GNN model in your research, please cite the following:

    @inproceedings{Wu:2019vb,
    author = {Wu, Shu and Tang, Yuyuan and Zhu, Yanqiao and Wang, Liang and Xie, Xing and Tan, Tieniu},
	title = {Session-based Recommendation with Graph Neural Networks},
	booktitle = {Proceedings of The Twenty-Third AAAI Conference on Artificial Intelligence},
	series = {AAAI '19},
	year = {2019},
	url = {http://arxiv.org/abs/1811.00855}
    }
