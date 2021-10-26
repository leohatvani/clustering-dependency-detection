Automated Functional Dependency Detection from Natural Language Test Specifications and Empirical Evaluation
=======

### Warning
This is an academic prototype software and the authors are not responsible for any damage resulting from its use.

### Installation
```bash
clone this repository
cd clustering-dependency-detection

# initiate a virtual environment
python3 -m venv env
source env/bin/activate
# upgrade pip if needed
pip install --upgrade pip
# install the required libraries
pip install -r requirements.txt

### Requirement
numpy==1.14.3
astroid==1.6.4
cycler==0.10.0
decorator==4.3.0
hdbscan==0.8.13
isort==4.3.4
kiwisolver==1.0.1
lazy-object-proxy==1.3.1
matplotlib==2.2.2
mccabe==0.6.1
networkx==2.1
pandas==0.23.0
pylint==1.9.1
pyparsing==2.2.0
python-dateutil==2.7.3
pytz==2018.4
scikit-fuzzy==0.3.1
scikit-learn==0.19.1
scipy==1.1.0
seaborn==0.8.1
six==1.11.0
sklearn==0.0
wrapt==1.10.11

### Packages
conda install -c conda-forge scikit-fuzzy
conda install -c conda-forge hdbscan

pip install -U scikit-fuzzy
pip install hdbscan

sudo apt-get install python3-tk
or
sudo dnf install python3-tkinter

try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter 
except ImportError:
    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here
    

### Feature Vector Generation
First transform your input documents into the CSV file format:
```csv
text,label
"This is the text of the input document",TC0000
...
```
Let's call this file `input.csv`.

Then, to generate the feature vector file, follow the steps for the installation and use of [Paragraph Vectors](https://github.com/inejc/paragraph-vectors). Train Paragraph Vectors using a command similar to:
```bash
python train.py start --data_file_name input.csv --num_epochs 100 \
       --num_noise_words 2 --vec_dim 64 --batch_size 32 --lr 1e-3
```

Finally, export the feature vectors into the CSV file format:
```bash
python export_vectors.py start --data_file_name input.csv --model_file_name input_model.xxxxxx.pth.tar
```
where xxxxxx should be replaced according to the results of running the Paragraph Vectors implementation.



### Running
Together with our code, we have included anonymized data that can be used to recreate the results that we have presented in our paper. There are two files `dataset_graph.csv` containing the description of the dependency graph between the test cases and the requirements and `dataset_vec.csv` containing the feature vectors of the input documents.

While in the virtual environment, to cluster using HDBSCAN:
```bash
python cluster.py --vectors dataset_vectors.csv --dependencies dataset_graph.csv
```

While in the virtual environment, to cluster using Fuzzy C-means:
```bash
python cluster.py --vectors dataset_vectors.csv --dependencies dataset_graph.csv --method fcm --nclusters 45
```
Replace 45 with the desired number of clusters.

The output will be stored in the `results` subfolder which will be created if it is missing.

### Authors

[Leo Hatvani](https://twitter.com/leo8)

[Sahar Tahvili](https://twitter.com/sahartahvili)

