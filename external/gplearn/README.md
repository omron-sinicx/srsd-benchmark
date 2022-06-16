# GP-based Baseline

## Setup pipenv
```shell
pipenv install --python 3.7
```

## Run experiments
```shell
for srsd_category in easy medium hard; do
  echo "[SRSD category: ${srsd_category}]"
  for filepath in ../../resource/datasets/srsd-feynman_${srsd_category}/train/*; do
    echo "[Current file: ${filepath}]"
    PARENT_DIR=$(dirname $(dirname $filepath))
    FILE_NAME=$(basename $filepath)
    TRAIN_FILE=${PARENT_DIR}/train/${FILE_NAME}
    VAL_FILE=${PARENT_DIR}/val/${FILE_NAME}
    TEST_FILE=${PARENT_DIR}/test/${FILE_NAME}
    pipenv run python gp_runner.py --train ${TRAIN_FILE} --val ${VAL_FILE} --test ${TEST_FILE} --config configs/optuna.yaml --out ${FILE_NAME}
  done
done
```
