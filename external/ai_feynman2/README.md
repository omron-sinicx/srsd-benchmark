# AI Feynman baseline

## Setup Docker / Singularity
```shell
docker build -t ai_feynman2 .

docker run --gpus all \
    -v ../../resource/datasets/:/resource/dataset/:ro \
    -v ../../resource/experiments/:/resource/experiments/:rw \
    -it ai_feynman2 /bin/bash
```

or Singularity
```shell
singularity build --fakeroot ai_feynman2.sif ai_feynman2.def
```

## Run experiments

### Docker user
```shell
mkdir case{1..10}/
cp ai_feynman_runner.py case{1..10}/

for srsd_category in easy medium hard; do
  echo "[SRSD category: ${srsd_category}]"
  for filepath in /resource/datasets/srsd-feynman_${srsd_category}/train/*; do
    echo "[Current file: ${filepath}]"
    cd ./case1/
    python ai_feynman_runner.py --src ${filepath} --op 7ops.txt --epoch 300
    cd ../case2/
    python ai_feynman_runner.py --src ${filepath} --op 10ops.txt --epoch 300
    cd ../case3/
    python ai_feynman_runner.py --src ${filepath} --op 14ops.txt --epoch 300
    cd ../case4/
    python ai_feynman_runner.py --src ${filepath} --op 19ops.txt --epoch 300
    cd ../case5/
    python ai_feynman_runner.py --src ${filepath} --op 7ops.txt --epoch 500
    cd ../case6/
    python ai_feynman_runner.py --src ${filepath} --op 10ops.txt --epoch 500
    cd ../case7/
    python ai_feynman_runner.py --src ${filepath} --op 14ops.txt --epoch 500
    cd ../case8/
    python ai_feynman_runner.py --src ${filepath} --op 19ops.txt --epoch 500
    cd ../case9/
    python ai_feynman_runner.py --src ${filepath} --op 14ops.txt --epoch 300 --bftt 120 --poly_deg 4
    cd ../case10/
    python ai_feynman_runner.py --src ${filepath} --op 19ops.txt --epoch 300 --bftt 120 --poly_deg 4
    cd ../
  done
done
```


## Convert equations to sympy objects
```shell
for i in {1..10}; do
  python equation_converter.py --solution case${i}/results/solution_feynman- --out all_results/case${i}/
  rename 's/solution_//' all_results/case${i}/*
done
```