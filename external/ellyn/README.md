# Ellyn-based Baseline (AFP and AFP-FE)

## Setup Docker / Singularity
```shell
docker build -t ellyn/latest .
```
or Singularity
```shell
singularity build --fakeroot ellyn.sif ellyn.def
```

## Run experiments with Docker
```shell
mkdir ../../resource/experiments/ -p
docker run --runtime=nvidia --gpus all \
    -v ../../resource/datasets/:/resource/datasets/:ro \
    -v ../../resource/experiments/:/resource/experiments/:rw \
    -it ellyn /bin/bash

conda activate ellyn-env
```

### Training batch job without fitness prediction
```shell
for srsd_category in easy medium hard; do
  echo "[SRSD category: ${srsd_category}]"
  for filepath in /resource/datasets/srsd-feynman_${srsd_category}/train/*; do
    echo "[Current file: ${filepath}]"
    PARENT_DIR=$(dirname $(dirname $filepath))
    FILE_NAME=$(basename $filepath)
    TRAIN_FILE=${PARENT_DIR}/train/${FILE_NAME}
    VAL_FILE=${PARENT_DIR}/val/${FILE_NAME}
    TEST_FILE=${PARENT_DIR}/test/${FILE_NAME}
    python ellyn_runner.py --train ${TRAIN_FILE} --val ${VAL_FILE} --test ${TEST_FILE} --config configs/afp_optuna.yaml --out ${FILE_NAME}
  done
done
```

### Training batch job with fitness prediction
```shell
for srsd_category in easy medium hard; do
  echo "[SRSD category: ${srsd_category}]"
  for filepath in /resource/datasets/srsd-feynman_${srsd_category}/train/*; do
    echo "[Current file: ${filepath}]"
    PARENT_DIR=$(dirname $(dirname $filepath))
    FILE_NAME=$(basename $filepath)
    TRAIN_FILE=${PARENT_DIR}/train/${FILE_NAME}
    VAL_FILE=${PARENT_DIR}/val/${FILE_NAME}
    TEST_FILE=${PARENT_DIR}/test/${FILE_NAME}
    python ellyn_runner.py --train ${TRAIN_FILE} --val ${VAL_FILE} --test ${TEST_FILE} --config configs/fe_afp_optuna.yaml --out ${FILE_NAME}
  done
done
```


## Run experiments with Singularity

### Training batch job without fitness prediction
```shell
for srsd_category in easy medium hard; do
  echo "[SRSD category: ${srsd_category}]"
  for filepath in ./resource/datasets/srsd-feynman_${srsd_category}/train/*; do
    echo "[Current file: ${filepath}]"
    PARENT_DIR=$(dirname $(dirname $filepath))
    FILE_NAME=$(basename $filepath)
    TRAIN_FILE=${PARENT_DIR}/train/${FILE_NAME}
    VAL_FILE=${PARENT_DIR}/val/${FILE_NAME}
    TEST_FILE=${PARENT_DIR}/test/${FILE_NAME}
    singularity exec --nv ellyn.sif /opt/conda/envs/ellyn-env/bin/python ellyn_runner.py --train ${TRAIN_FILE} --val ${VAL_FILE} --test ${TEST_FILE} --config configs/afp_optuna.yaml --out ${FILE_NAME}
  done
done
```

### Training batch job with fitness prediction
```shell
for srsd_category in easy medium hard; do
  echo "[SRSD category: ${srsd_category}]"
  for filepath in ./resource/datasets/srsd-feynman_${srsd_category}/train/*; do
    echo "[Current file: ${filepath}]"
    PARENT_DIR=$(dirname $(dirname $filepath))
    FILE_NAME=$(basename $filepath)
    TRAIN_FILE=${PARENT_DIR}/train/${FILE_NAME}
    VAL_FILE=${PARENT_DIR}/val/${FILE_NAME}
    TEST_FILE=${PARENT_DIR}/test/${FILE_NAME}
    singularity exec --nv ellyn.sif /opt/conda/envs/ellyn-env/bin/python ellyn_runner.py --train ${TRAIN_FILE} --val ${VAL_FILE} --test ${TEST_FILE} --config configs/fe_afp_optuna.yaml --out ${FILE_NAME}
  done
done
```
