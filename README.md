Copyright (C) 2022 Rudolf Olah

Datasets are prefixed with `dataset_`

## Setup

Setup virtualenv and install dependencies:

```shell
sudo pip install virtualenv
virtualenv --python=python3.9 env
source env/bin/activate
pip install -r requirements.txt
```

## Run

```shell
source env/bin/activate
python main.py
```

GPU not working? Check this for adding GPU support to TensorFlow: https://www.tensorflow.org/install/gpu
