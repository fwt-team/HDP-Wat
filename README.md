# Code for the 'Unsupervised Grouped Axial Data Modeling via Hierarchical Bayesian Nonparametric Models with Watson Distributions'

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
conda activate Test
```

## File

    datas              # container of data  
    config.py          # the hyper parameters of dataset  
    model.py           # hdp-wat and hpyp-wat model code  
    utils.py           # some util functions  
    train_synthetic.py # training code of synthetic data  
    train_gene.py      # training code of gene data  
    train_nyu.py       # training code of deep image data  

## Training

To train the model(s) in the paper, run this command:  

    __params:__  
    -name     dataset name  
    -lp       Load hyper parameter or not 
    -verbose  print information or not  

    -k        first truncation of model  
    -t        second truncation of model  
    -tau      stick hyper params of fist level  
    -gamma    stick hyper params of second level  
    -th       second level threshold of converge   
    -mth      the threshold of Cluster number  
    -sm       second level max iteration  
    -m        max iterations of training  

```train
python train_synthetic.py -name syn_data1 -lp 1 -verbose 1 -k 10 -t 5 -tau 1 -gamma 0.01 -th 1e-7 -mth 0.01 -sm -1 -m 20
or
python train_gene.py -name Sporulation -lp 1 -verbose 1 -k 20 -t 20 -tau 20 -gamma 0.01 -th 1e-7 -mth 0.01 -sm 500 -m 10
or
python train_nyu.py -name nyu -lp 1 -verbose 1 -k 20 -t 10 -tau 20 -gamma 0.01 -th 1e-7 -mth 0.01 -sm 1000 -m 10
```

---
### Reference
If you use our code in your work, please cite our paper. 

    @article{FAN2021,
    title = "Unsupervised Grouped Axial Data Modeling via Hierarchical Bayesian Nonparametric Models with Watson Distributions",
    author = "Wentao Fan and Lin Yang and Nizar Bouguila",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    volume = "0",
    pages = "1",
    year = "2021"
    }

