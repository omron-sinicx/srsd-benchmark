# Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery
This is the official code repository of ["Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery"](https://github.com/omron-sinicx/srsd-benchmark#citation). 
This work revisits datasets and evaluation criteria for Symbolic Regression (SR), specifically focused on its potential 
for scientific discovery. Focused on a set of formulas used in the existing datasets based on 
Feynman Lectures on Physics, we recreate 120 datasets to discuss the performance of symbolic regression for 
scientific discovery (SRSD). For each of the 120 SRSD datasets, we carefully review the properties of the formula and 
its variables to design reasonably realistic sampling ranges of values so that our new SRSD datasets can be used for 
evaluating the potential of SRSD such as whether or not an SR method can (re)discover physical laws from such datasets. 
We also create another 120 datasets that contain dummy variables to examine whether SR methods can choose necessary 
variables only. Besides, we propose to use normalized edit distances (NED) between a predicted equation and the true 
equation trees for addressing a critical issue that existing SR metrics are either binary or errors between the target 
values and an SR model’s predicted values for a given input. We conduct experiments on our new SRSD datasets using six 
SR methods. The experimental results show that we provide a more realistic performance evaluation, and our user study 
shows that the NED correlates with human judges significantly more than an existing SR metric.

[![YouTube](https://img.youtube.com/vi/MmeOXuUUAW0/0.jpg)](https://www.youtube.com/watch?v=MmeOXuUUAW0)

## Setup 
We used pipenv for a Python virtual environment.

```shell
pipenv install --python 3.8
mkdir -p resource/datasets 
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
  
We also created another 120 SRSD datasets by introducing dummy variables to the 120 SRSD datasets.
- [Easy set w/ dummy variables](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy_dummy)
- [Medium set w/ dummy variables](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_medium_dummy)
- [Hard set w/ dummy variables](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard_dummy)
  
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


### Introduce dummy variables
To re-introduce dummy variables (columns) to the datasets, run the following commands:

```shell
pipenv run python dummy_column_mixer.py --input ./resource/datasets/srsd/easy_set/ --output ./resource/datasets/srsd/easy_set_dummy/
pipenv run python dummy_column_mixer.py --input ./resource/datasets/srsd/medium_set/ --output ./resource/datasets/srsd/medium_set_dummy/
pipenv run python dummy_column_mixer.py --input ./resource/datasets/srsd/hard_set/ --output ./resource/datasets/srsd/hard_set_dummy/
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
(If your model uses input variable index starting from 1 like DSO, you can still use the expressions by adding `-dec_idx` to the following command.)
e.g., DSO

```shell
pipenv run python model_selector.py --est ./dso_results/est_eq* \
    --val ~/dataset/symbolic_regression/srsd-feynman_all/val/ \
    --output ./results/dso_models/ \
    -dec_idx
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

## Citation
[[OpenReview](https://openreview.net/forum?id=qrUdrXsiXX)] [[Video](https://www.youtube.com/watch?v=MmeOXuUUAW0)] [[Preprint](https://arxiv.org/abs/2206.10540)]  
```bibtex
@article{matsubara2024rethinking,
  title={Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery},
  author={Matsubara, Yoshitomo and Chiba, Naoya and Igarashi, Ryo and Ushiku, Yoshitaka},
  journal={Journal of Data-centric Machine Learning Research},
  year={2024},
  url={https://openreview.net/forum?id=qrUdrXsiXX}
}
```
