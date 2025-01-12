# Environment setup

Mac
```
brew install pyenv
```

Linux
```
pyenv update
curl https://pyenv.run | bash
# Follow the instructions and modify ~/.bashrc
source ~/.bashrc
```

All
```
pyenv install -v 3.8
pyenv local 3.8
python -m venv .venv
source .venv/bin/activate
cd session-1/
pip install -r requirements.txt
pip install --upgrade pip
python main.py 
```

## Dataset

```
pip install kaggle
mkdir -p datasets
# Add datasets to .gitginore


# Upload kaggle.json to /home/gitpod/.kaggle

# For GitPod
mkdir -p /home/gitpod/.kaggle
mv kaggle.json /home/gitpod/.kaggle
chmod 600 /home/gitpod/.kaggle/kaggle.json

# For GitHub
mkdir -p /home/codespace/.kaggle
mv kaggle.json /home/codespace/.kaggle
chmod 600 /home/codespace/.kaggle/kaggle.json

# For local
mkdir -p /Users/jcajidefernandez/.kaggle
mv kaggle.json /Users/jcajidefernandez/.kaggle
chmod 600 /Users/jcajidefernandez/.kaggle/kaggle.json

kaggle datasets download -d gpreda/chinese-mnist -p datasets
cd datasets
unzip chinese-mnist.zip    
```

```
cd session-2
pip install -r requirements.txt
pip install pandas
pip install torchvision
pip install scikit-learn
pip install seaborn
pip install -U "ray[data,train,tune,serve]"
pip install tensorboard
```

## Session 5
```
cd session-5-dev

brew install xz
pyenv versions
# https://stackoverflow.com/questions/59690698/modulenotfounderror-no-module-named-lzma-when-building-python-using-pyenv-on
pyenv uninstall 3.10.6
pyenv install 3.10.6

# Specific env
pyenv local 3.10.6
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Execution**
```
tensorboard --logdir=logs
python main.py --task reconstruction --log_framework tensorboard
python main.py --task classification --log_framework tensorboard
```

## Session 6
```
docker build -t session6 .
podman build -t session6 --platform linux/amd64 .

podman run -v

podman ps

podman exec -it session6 /bin/bash

podman run -v $(pwd)/data:/data -v $(pwd)/checkpoints:/checkpoints -it session6 train 

podman run -v $(pwd)/data/:/data -v $(pwd)/checkpoints:/checkpoints -it session6 predict 0.24522,0,9.9,0,0.544,5.782,71.7,4.0317,4,304,18.4,396.9,15.94
```