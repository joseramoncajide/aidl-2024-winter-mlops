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