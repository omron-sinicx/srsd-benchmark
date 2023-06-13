# End-to-End Symbolic Regression with Transformers

```shell
pipenv install
git clone https://github.com/facebookresearch/symbolicregression org
mv org/* ./
rm org -rf
mkdir resource/ckpt -p
wget https://dl.fbaipublicfiles.com/symbolicregression/model1.pt
mv model1.pt resource/ckpt/model.pt
```

### Training batch job
```
CKPT_FILE=./resource/ckpt/model.pt
for group_name in easy medium hard; do
    OUT_DIR=./e2e-sr_w_transformer-results/srsd-feynman_${group_name}
    mkdir ${OUT_DIR} -p
    for filepath in ~/dataset/symbolic_regression/srsd-feynman_${group_name}/train/*; do
        PARENT_DIR=$(dirname $(dirname $filepath))
        FILE_NAME=$(basename $filepath)
        TRAIN_FILE=${PARENT_DIR}/train/${FILE_NAME}
        TEST_FILE=${PARENT_DIR}/test/${FILE_NAME}
        timeout 5m pipenv run python runner.py --train ${TRAIN_FILE} --test ${TEST_FILE} --ckpt ${CKPT_FILE} --out ${OUT_DIR}/${FILE_NAME}
    done
done
```

