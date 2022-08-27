# Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery
This is the official code repository of ["Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery"](https://github.com/omron-sinicx/srsd-benchmark#citation). 
This work revisits datasets and evaluation criteria for Symbolic Regression, a task of expressing given data 
using mathematical equations, specifically focused on its potential for scientific discovery. 
Focused on a set of formulas used in the existing datasets based on Feynman Lectures on Physics, 
we recreate 120 datasets to discuss the performance of symbolic regression for scientific discovery (SRSD). 
For each of the 120 SRSD datasets, we carefully review the properties of the formula and its variables 
to design reasonably realistic sampling range of values so that our new SRSD datasets can be used for 
evaluating the potential of SRSD such as whether or not an SR method can (re)discover physical laws from such datasets. 
As an evaluation metric, we also propose to use normalized edit distances between a predicted equation 
and the ground-truth equation trees. While existing metrics are either binary or errors between the target values and 
an SR model's predicted values for a given input, normalized edit distances evaluate a sort of similarity between 
the ground-truth and predicted equation trees. We have conducted experiments on our new SRSD datasets using five 
state-of-the-art SR methods in SRBench and a simple baseline based on a recent Transformer architecture. 
The results show that we provide a more realistic performance evaluation and open up a new machine learning-based 
approach for scientific discovery.

## Setup 
We used pipenv for a Python virtual environment.

```shell
pipenv install --python 3.7
mkdir resource/ 
```

You can also use conda or local Python environments.
Use `requirements.txt` if you use either conda or local Python environments.
e.g., 
```shell
# Activate your conda environment if you use conda, and run the following commands
pip install pip --upgrade
pip install -r requirements.txt
```

## Download or re-generate SRSD datasets
Our SRSD datasets are publicly available at Hugging Face Dataset repositories:
- [Easy set](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy)
- [Medium set](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium)
- [Hard set](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard)
  
Download and store the datasets at `./resource/datasets/`

If you want to re-generate the SRSD datasets,
```shell
pipenv run python dataset_generator.py --config configs/datasets/feynman/easy_set.yaml
pipenv run python dataset_generator.py --config configs/datasets/feynman/medium_set.yaml
pipenv run python dataset_generator.py --config configs/datasets/feynman/hard_set.yaml
```

Note that the resulting datasets may not exactly match those we published as the values are resampled from the same distributions.

Also, run the following command for merging all the sets
```shell
cp ./resource/datasets/srsd-feynman_easy/ ./resource/datasets/srsd-feynman_all/ -r
cp ./resource/datasets/srsd-feynman_medium/ ./resource/datasets/srsd-feynman_all/ -r
cp ./resource/datasets/srsd-feynman_hard/ ./resource/datasets/srsd-feynman_all/ -r
```

### Convert SRSD datasets for DSR
```shell
for srsd_category in easy medium hard; do
  for split_name in train val test; do
    pipenv run python dataset_converter.py --src ./resource/datasets/srsd-feynman_${srsd_category}/${split_name}/ \
      --dst ./resource/datasets/srsd-feynman_${srsd_category}_csv/${split_name}/ --dst_ext .csv
  done
  cp ./resource/datasets/srsd-feynman_${srsd_category}/true_eq/ ./resource/datasets/srsd-feynman_${srsd_category}_csv/true_eq/ -r
done
```

## Run existing symbolic regression baselines / your own model
Follow the instruction in [external/README.md](./external)

## Analyze equations

### SRSD-Feynman datasets
Check equation properties (e.g., number of variables and those used in sympy equation)
```shell
pipenv run python eq_analyzer.py --name feynman -simple_check
```

Visualize equation trees
```shell
pipenv run python eq_analyzer.py --name feynman -visualize --output ./eq_trees/
```


## Select the best model per dataset for DSO and AI Feynman
Due to their original implementations, DSO and AI Feynman are difficult to work with optuna for hyperparameter tuning.
If there are multiple trained models with different seeds and/or hyperparameters, select the best model per dataset 
based on relative error on validation split like other methods in this repository.  
e.g., DSO

```shell
pipenv run python model_selector.py --est ./dso_results/est_eq* \
    --val ~/dataset/symbolic_regression/srsd-feynman_all/val/ \
    --output ./results/dso_models/
```

## Compute edit distance between estimated and ground-truth equations
Both the estimated and ground-truth equations need to be "pickled" as sympy equations e.g., [sympy.sympify(eq_str)](https://github.com/omron-sinicx/srsd-benchmark/blob/main/external/gplearn/gp_runner.py#L83-L85) and [pickle the object](https://github.com/omron-sinicx/srsd-benchmark/blob/main/external/gplearn/gp_runner.py#L77-L80)  
Mathematical operations available in [sympy](https://www.sympy.org/en/index.html) are supported, 
and input variables should use sympy.Symbol(f'x{index}'), where `index` is integer and starts from 0.  
(If your model uses input variable index starting from 1 like DSO, you can still use the expressions by adding `-dec_idx` to the following command.)

1 by 1
```shell
pipenv run python eq_comparator.py \
    --est external/gplearn/gplearn_w_optuna/feynman-i.12.1-est_eq.pkl \
    --gt ./resource/datasets/srsd-feynman_easy/true_eq/feynman-i.12.1.pkl
```

Batch process
```shell
pipenv run python eq_comparator.py \
    --method_name gplearn \
    --est external/gplearn/gplearn_w_optuna/ \
    --est_delim .txt-est_eq \
    --gt ./resource/datasets/srsd-feynman_easy/true_eq/ \
    --gt_delim .pkl \
    --eq_table feynman_eq.tsv \
    --dist_table feynman_dist.tsv \
    -normalize
```

Add `-dec_idx` for DSO's estimated equations to decrement variable indices since DSR's variable indices start from 1 instead of 0.


## Generate random equation datasets
Run `random_dataset_generator.ipynb`  

## Compare SRSD datasets with randomly generated datasets
```shell
pipenv run python dataset_comparator.py \
  --src_eq ~/dataset/symbolic_regression/srsd-feynman_all/true_eq/ \
  --src_tabular ~/dataset/symbolic_regression/srsd-feynman_all/test/ \
  --dst_eq ~/dataset/symbolic_regression/bigram_nb-feynman_like_random_set/true_eq/ \
  --dst_tabular ~/dataset/symbolic_regression/bigram_nb-feynman_like_random_set/train/
```


## Symbolic Transformer baseline

### Pretraining
You can skip this pretraining step if you use [our pretrained model checkpoint](https://github.com/omron-sinicx/srsd-benchmark/releases/tag/v0.1.0).

```shell
pipenv run python symbolic-regression-for-mis/symbolic_regression.py --config ./configs/experiments/colab_pro/symbolic_transformer-pretraining.yaml --log ./symbolic-regression-for-mis/log/experiments/colab_pro/symbolic_transformer-pretraining.txt
```

### Normalized edit distance evaluation
```shell

pipenv run python symbolic_regression.py -test_only --config ./experiments/symbolic_transformer-srsd-feynman_easy.yaml --log ./log/experiments/symbolic_transformer-srsd-feynman_easy.txt
pipenv run python symbolic_regression.py -test_only --config ./experiments/symbolic_transformer-srsd-feynman_medium.yaml --log ./log/experiments/symbolic_transformer-srsd-feynman_medium.txt
pipenv run python symbolic_regression.py -test_only --config ./experiments/symbolic_transformer-srsd-feynman_hard.yaml --log ./log/experiments/symbolic_transformer-srsd-feynman_hard.txt
```

### R2-based accuracy evaluation
```shell
# Easy set
pipenv run python symbolic_regression.py -test_only -estimate_coeff \
  --config ./experiments/symbolic_transformer-srsd-feynman_easy.yaml \
  --log ./log/experiments_estimate_coeff/symbolic_transformer-srsd-feynman_easy.txt
pipenv run python r2_evaluator.py \
  --est ./resource/pred_pickles/srsd-feynman_easy/ \
  --test ./resource/datasets/srsd-feynman_easy/test/ \
  --est_delim .pkl \
  --test_delim .txt \
  --r2thr 0.999

# Medium set
pipenv run python symbolic_regression.py -test_only -estimate_coeff \
  --config ./experiments/symbolic_transformer-srsd-feynman_medium.yaml \
  --log ./log/experiments_estimate_coeff/symbolic_transformer-srsd-feynman_medium.txt
pipenv run python r2_evaluator.py \
  --est ./resource/pred_pickles/srsd-feynman_medium/ \
  --test ./resource/datasets/srsd-feynman_medium/test/ \
  --est_delim .pkl \
  --test_delim .txt \
  --r2thr 0.999

# Hard set
pipenv run python symbolic_regression.py -test_only -estimate_coeff \
  --config ./experiments/symbolic_transformer-srsd-feynman_hard.yaml \
  --log ./log/experiments_estimate_coeff/symbolic_transformer-srsd-feynman_hard.txt
pipenv run python r2_evaluator.py \
  --est ./resource/pred_pickles/srsd-feynman_hard/ \
  --test ./resource/datasets/srsd-feynman_hard/test/ \
  --est_delim .pkl \
  --test_delim .txt \
  --r2thr 0.999
```

## Citation
[[Preprint](https://arxiv.org/abs/2206.10540)]  
```bibtex
@article{matsubara2022rethinking,
  title={Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery},
  author={Matsubara, Yoshitomo and Chiba, Naoya and Igarashi, Ryo and Tatsunori, Taniai and Ushiku, Yoshitaka},
  journal={arXiv preprint arXiv:2206.10540},
  year={2022}
}
```
