# Causally Disentangled Generative Variational AutoEncoder

This repository is the official implementation of Causally Disentangled Generative Variational AutoEncoder with pytorch. 

## Package Dependencies

```setup
python==3.7
numpy==1.21.6
torch==1.13.0
```
Additional package requirements for this repository are described in `requirements.txt`.

## Training & Evaluation 

### 1. How to Training & Evaluation  

#### 1. pendulum datset experiment

- training CDG-VAE
```
python main.py --model "GAM"
```   
- training CDG-VAE under semi-supervised learning
```
python main_semi.py --model "GAM"
```   
- training CDG-VAE for evaluation of distributional robustness
```
python DR/main.py --model "GAM"
```   
- training CDG-VAE for evaluation of distributional robustness under semi-supervised learning
```
python DR/main_semi.py --model "GAM"
```   

#### 1-1. pendulum dataset results
- counterfactual image generation
```
python inference.py
```
- sample efficiency
```
python sample_efficiency.py
```
- distributional robustness
```
python DR/robustness.py
```

#### 2. tabular dataset experiment
- training CDG-VAE
```
python tabular/main.py --model "GAM"
```  
- training CDG-TVAE
```
python tabular/main_tvae.py 
```  

#### 2-1. tabular dataset results
- for CDG-VAE
```
python tabular/inference.py
```  
- for CDG-TVAE
```
python tabular/inference_tvae.py 
```  

<!-- ## Results

<center><img  src="https://github.com/an-seunghwan/causal_vae/blob/main/assets/do/do_GAMsemi_nonlinear.png?raw=true" width="800"  height="400"></center> -->

### 1. directory and codes

```
.
+-- assets 
|       +-- CDM 
|       +-- do (counterfactual images)
|       +-- sample_efficiency

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
+-- requirements.txt
+-- LICENSE
+-- README.md
+-- sweep.py
+-- sweep.yaml

+-- DR (folder which contains source codes for distributional robustness experiments)
|   +-- assets 
|       +-- robustness
|   +-- main.py
|   +-- main_semi.py
|   +-- robustness.py
|   +-- toy_DR.py

+-- tabular (folder which contains source codes for tabular dataset experiments)
|   +-- assets 
|       +-- adult
|       +-- covtype
|       +-- loan
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
|   +-- sweep.py
|   +-- sweep.yaml
```