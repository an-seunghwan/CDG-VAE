# Causally Disentangled Generative Variational AutoEncoder

This repository is the official implementation of Causally Disentangled Generative Variational AutoEncoder (26th European Conference on Artificial Intelligence ECAI 2023 accepted paper) with pytorch. 

> **_NOTE:_** This repository supports [WandB](https://wandb.ai/site) MLOps platform!

## Appendix
See `appendix.pdf` for the appendix file for the main manuscript.

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
+-- appendix.pdf

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

## Citation
```
@inproceedings{DBLP:conf/ecai/AnSJ23,
  author       = {SeungHwan An and
                  Kyungwoo Song and
                  Jong{-}June Jeon},
  editor       = {Kobi Gal and
                  Ann Now{\'{e}} and
                  Grzegorz J. Nalepa and
                  Roy Fairstein and
                  Roxana Radulescu},
  title        = {Causally Disentangled Generative Variational AutoEncoder},
  booktitle    = {{ECAI} 2023 - 26th European Conference on Artificial Intelligence,
                  September 30 - October 4, 2023, Krak{\'{o}}w, Poland - Including
                  12th Conference on Prestigious Applications of Intelligent Systems
                  {(PAIS} 2023)},
  series       = {Frontiers in Artificial Intelligence and Applications},
  volume       = {372},
  pages        = {93--100},
  publisher    = {{IOS} Press},
  year         = {2023},
  url          = {https://doi.org/10.3233/FAIA230258},
  doi          = {10.3233/FAIA230258},
  timestamp    = {Wed, 18 Oct 2023 09:31:16 +0200},
  biburl       = {https://dblp.org/rec/conf/ecai/AnSJ23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
