# Environment setup

```
brew install pyenv
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
mkdir -p /home/gitpod/.kaggle
mv kaggle.json /home/gitpod/.kaggle
chmod 600 /home/gitpod/.kaggle/kaggle.json

kaggle datasets download -d gpreda/chinese-mnist -p datasets
cd datasets
unzip chinese-mnist.zip    
```

```
cd session-2
pip install -r requirements.txt
pip install pandas
pip install torchvision
```