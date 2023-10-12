# Discovering Symbolic Models from Deep Learning with Inductive Biases

```shell
conda create -n pysr python=3.10
conda activate pysr
conda install -c conda-forge pyjuliaup
conda install -c conda-forge pysr
python3 -m pysr install
```

### Training batch job
```
for group_name in easy medium hard; do
    RESULT_DIR=./results/srsd-feynman_${group_name}
    EQ_DIR=./eqs/srsd-feynman_${group_name}
    mkdir ${RESULT_DIR} -p
    mkdir ${EQ_DIR} -p
    for filepath in ~/dataset/symbolic_regression/srsd-feynman_${group_name}/train/*; do
        PARENT_DIR=$(dirname $(dirname $filepath))
        FILE_NAME=$(basename $filepath)
        TRAIN_FILE=${PARENT_DIR}/train/${FILE_NAME}
        python3 runner.py --config ./config.yaml --train ${TRAIN_FILE} --eq ${EQ_DIR}/${FILE_NAME}.pkl --table ${RESULT_DIR}/${FILE_NAME}.tsv
    done
done
```

