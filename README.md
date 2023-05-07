# Causally Disentangled Generative Variational AutoEncoder

This repository is the official implementation of Causally Disentangled Generative Variational AutoEncoder (CDG-VAE) with pytorch. 

> **_NOTE:_** This repository supports [WandB](https://wandb.ai/site) MLOps platform!

## Training & Evaluation 

### 1. How to Training & Evaluation  

#### 1. pendulum datset

#### training 

- training CDG-VAE
```
python main.py --model "CDGVAE"
```   
- training CDG-VAE in order to evaluate distributional robustness of downstream task
```
python DR/main.py --model "CDGVAE"
```   
(Note) The file names with `_semi` means training CDG-VAE under semi-supervised learning setting.

#### evaluation
- counterfactual image generation: `inference.py`
- sample efficiency of downstream task: `sample_efficiency.py`
- distributional robustness of downstream task: `DR/robustness.py`

#### 2. tabular dataset 

#### training 
- CDG-VAE
```
python tabular/main.py --model "CDGVAE"
```  
- CDG-TVAE
```
python tabular/main_tvae.py 
```  

#### evaluation (SHD and synthetic data quality)
- CDG-VAE: `tabular/inference.py`
- CDG-TVAE: `tabular/inference_tvae.py `

## Results

<center><img  src="https://github.com/an-seunghwan/causal_vae/blob/main/assets/cdgvae_github.png?raw=true" width="800"  height="350"></center>

## directory and codes

```
.
+-- assets 
+-- modules 
|       +-- datasets.py
|       +-- model.py
|       +-- pendulum_real.py
|       +-- pendulum.py
|       +-- simulation.py
|       +-- train.py
|       +-- viz.py

+-- main.py
+-- main_semi.py
+-- main_classifier.py
+-- inference.py
+-- metric.py
+-- sample_efficiency.py
+-- LICENSE
+-- README.md

+-- DR (folder which contains source codes for distributional robustness experiments)
|   +-- assets 
|   +-- main.py
|   +-- main_semi.py
|   +-- robustness.py
|   +-- toy_DR.py

+-- tabular (folder which contains source codes for tabular dataset experiments)
|   +-- assets 
|   +-- modules
|       +-- adult_datasets.py
|       +-- covtype_datasets.py
|       +-- loan_datasets.py
|       +-- errors.py
|       +-- numerical.py
|       +-- transformer_base.py
|       +-- transformer_null.py
|       +-- data_transformer.py
|       +-- model.py
|       +-- train.py
|       +-- evaluation.py
|       +-- viz.py
|       +-- simulation.py
|   +-- dag_adult.py
|   +-- dag_covtype.py
|   +-- dag_loan.py
|   +-- main.py
|   +-- inference.py
|   +-- main_tvae.py
|   +-- inference_tvae.py
```