# Deep Symbolic Regression baseline

## Setup pipenv
```shell
pipenv install --python 3.8
```

## Setup Docker / Singularity
```shell
docker build -t tf-dso .

DATASET_DIR=/host_disk/datasets
EXP_DIR=/host_disk/experiments
docker run --gpus all \
    -v ../../resource/datasets/:${DATASET_DIR}/:ro \
    -v ../../resource/experiments/:${EXP_DIR}/:rw \
    -it tf-dso /bin/bash
```

or Singularity
```shell
singularity build --fakeroot dso.sif dso.def
```

## Run experiments

### Docker user

DSR  
```shell
for srsd_category in easy medium hard; do
  echo "[SRSD category: ${srsd_category}]"
  for filepath in /resource/datasets/srsd-feynman_${srsd_category}/train/*; do
    echo "[Current file: ${filepath}]"
    for i in {1..5}; do
	  python -m dso.run ./configs/config_wo_const${i}.json --b ${filepath} --seed ${i}
	  python -m dso.run ./configs/config_w_const${i}.json --b ${filepath} --seed ${i}
	done
  done
done
```

uDSR  
```shell
for srsd_category in easy medium hard; do
  echo "[SRSD category: ${srsd_category}]"
  for filepath in /resource/datasets/srsd-feynman_${srsd_category}/train/*; do
    echo "[Current file: ${filepath}]"
    for i in {1..10}; do
	  python -m dso.run ./configs/config_w_poly{i}.json --b ${filepath} --seed ${i}
	done
  done
done
```

### Singularity user

DSR  
```shell
for srsd_category in easy medium hard; do
  echo "[SRSD category: ${srsd_category}]"
  for filepath in ./resource/datasets/srsd-feynman_${srsd_category}/train/*; do
    echo "[Current file: ${filepath}]"
    for i in {1..5}; do
	  singularity exec --nv ./dso.sif python -m dso.run ./configs/dsr/config_wo_const${i}.json --b ${filepath} --seed ${i}
	  singularity exec --nv ./dso.sif python -m dso.run ./configs/dsr/config_w_const${i}.json --b ${filepath} --seed ${i}
	done
  done
done
```

uDSR  
```shell
for srsd_category in easy medium hard; do
  echo "[SRSD category: ${srsd_category}]"
  for filepath in ./resource/datasets/srsd-feynman_${srsd_category}/train/*; do
    echo "[Current file: ${filepath}]"
    for i in {1..10}; do
	  singularity exec --nv ./dso.sif python -m dso.run ./configs/udsr/config_w_poly{i}.json --b ${filepath} --seed ${i}
	done
  done
done
```

## Extract and convert estimated equations

```shell
pipenv run python equation_converter.py \
    --summary ./logs_w_const/ \
    --out ./est_eq/
```
