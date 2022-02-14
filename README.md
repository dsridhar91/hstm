# Heterogeneous Supervised Topic Models (HSTMs)

**Introduction**

This repository contains the code and demo data for: 
- [1] _Heterogeneous Supervised Topic Models_, by Dhanya Sridhar, Hal Daumé III, and David Blei.

If you use this code please cite [1]. 

Researchers in the social sciences are often interested in the relationship between text and an outcome of interest, where the goal is to both uncover latent patterns in the text and predict outcomes for unseen texts. To this end, this paper develops the heterogeneous supervised topic models (HSTM), a probabilistic approach to text analysis and prediction. HSTMs posit a joint model of text and outcomes to find heterogeneous patterns that help with both text analysis and prediction. The main benefit of HSTMs is that they capture heterogeneity in the relationship between text and the outcome across latent topics. To fit HSTMs, we develop a variational inference algorithm based on the auto-encoding variational Bayes framework. We study the performance of HSTMs on eight datasets and find that they consistently outperform related methods, including fine-tuned black box models. Finally, we apply HSTMs to analyze news articles labeled with pro- or anti-tone. We find evidence of differing language used to signal a pro- and anti-tone.

**Requirements** 


The code has been tested on Python 3.6.9 with the following packages:
```bash
dill==0.2.8.2
nltk==3.4.5
numpy==1.18.1
pandas==0.24.1
scikit-learn==0.20.3
scipy==1.4.1
tensorflow==2.1.0
tensorflow-gpu==2.1.0
tensorflow-hub==0.7.0
tensorflow-probability==0.7.0
tokenizers==0.7.0
torch==1.3.1
torchvision==0.4.2
transformers==2.11.0
```

It is possible to install the dependencies using pip:
```bash
pip install -r requirements.txt
```

**Data**

We've included processed input files for the eight datasets on which we reported prediction error on Table 1 in the paper. The files are compressed Numpy files that include the document-term count matrix, and labels (referred to as outcomes in the paper). This is under:
```dat/proc/```

To run BERT, we've also included processed input files formatted as a CSV with the raw text and label as columns. This is under:
```dat/csv_proc/```

Soon, we'll provide instructions for downloading the unprocessed, raw datasets. Using Amazon as an example, raw data can be processed with our code by running
```bash
python -m data.dataset --data=amazon --data_file=../dat/reviews_Office_Products_5.json
``` 
assuming that the raw data files are stored where --data_file points.

**Reproducing predictive performance and ablation studies **

