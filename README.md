# Acoustic Scene Analysis With Multihead Self Attention
This repo contains implementation of the paper "Acoustic Scene Analysis With Multihead Self Attention" by Weimin Wang, Weiran Wang, Ming Sun, Chao Wang from Amazon Alexa team.

Paper: https://arxiv.org/pdf/1909.08961.pdf

## Installation

I suggest you to install Anaconda3 in your system. First download Anancoda3 from https://docs.anaconda.com/anaconda/install/hashes/lin-3-64/
```bash
bash Anaconda2-2019.03-Linux-x86_64.sh
```
## Clone the repo
```bash
git clone https://github.com/KrishnaDN/acoustic-scene-analysis-with-multihead-self-attention.git
```
Once you install anaconda3 successfully, install required packges using requirements.txt
```bash
pip install -r requirements.txt
```

## Data Processing and creating manifest files
To process DCASE2018 data. Download the dataset and use the following code. This script will create train.txt and eval.txt and places them in
'meta/' folder
```
python dataset/data_processing.py
```

## Training
To start the training using train.py. You can change the hyperparameters in the script. By default the code uses the hyperparameters mentioned in the paper 
```
python train.py
```

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
For any queries contact : krishnadn94@gmail.com
## License
[MIT](https://choosealicense.com/licenses/mit/)
