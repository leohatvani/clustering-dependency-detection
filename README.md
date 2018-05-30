Automated Functional Dependency Detection from Natural Language Test Specifications and Empirical Evaluation
=======

### Warning
This is an academic prototype software and the authors are not responsible for any damage resulting from its use.

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
```

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

