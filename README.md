# Heterogeneous Supervised Topic Models (HSTMs)

**Introduction**

This repository contains the code and demo data for: 
- [1] _Heterogeneous Supervised Topic Models_, by Dhanya Sridhar, Hal Daumé III, and David Blei.

If you use this code please cite [1]. 

**Abstract**

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

_Important note_: The experiments use Slurm (https://slurm.schedmd.com/documentation.html), a resource scheduler, to submit scripts to run on a server. The experiments are run on machines equipped with Titan GPUs and on average, 32GB of RAM. To use a different scheduler (or to run experiments locally), the scripts to submit experiments (described below) need to modified accordingly.

Finally, all the following commands must be run from the ```src/``` directory.

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

**Reproducing predictive performance and ablation studies**

To reproduce the results in tables 1 and 2 in the paper, execute the following steps.

First, run
```bash
./experiments/submit_scripts/submit_hyperparam.sh
```
This runs the experiments for HSTM and its variants (used in the ablation study). The STM variant is closely related to [3], when they use inferred topics
in a supervised model of labels. Many of the modeling choices are the same; one difference is that we initialize the log topics using LDA as we do with HSTMs, for a fair comparison.

Next, run
```bash
./experiments/submit_scripts/submit_bert.sh
```
This runs the fine-tuned BERT model for classification (or regression) [4]. We use the BERT model implemented here:
```https://huggingface.co/transformers/v2.11.0/model_doc/bert.html```
To make predictions with this model, we take the average of a sequence's final layer token embeddings, and apply a linear map to this per-sequence processed embedding to obtain either the logit prediction or the prediction.

Then, run
```bash
./experiments/submit_scripts/submit_slda.sh
```
This runs our auto-encoding variational Bayes implementation of supervised LDA [2].

The outputs from the experiment will saved under ```out/``` across several subdirectories. The details can be found by tracing through the submit scripts and ```src/experiment/run_experiment.py```.

Finally, run
```bash
jupyter notebook
```
and play through both ```run_baselines.ipynb``` and ```process_results.ipynb```. The first notebook executes the remaining baseline methods not listed above, and the second notebook visualizes the results for the baselines noted above.

**Fitting HSTMs for a new dataset**

This repository also supports fitting HSTMs for your own text data. It requires a few steps. 
As a running example, suppose your setting is called ```myDataset```.

First, you'll neeed to implement a function in ```data/load_data_from_text.py``` that processes your raw text and labels and returns two array objects,
```
tuple(np.array, np.array)
```
The first must be an array of texts (string) and the second is an array of labels. The other functions in the script can provide a template for processing your data. For consistency, it's recommmended that your function is called ```load_myDatset```.

_Note_: The code is currently written to support 0/1 labels and real-valued labels.

Second, you'll need to minimally modify ```data/dataset.py```. To the import statement on line [17], add your newly created function
```bash
from data.load_data_from_text import ..., load_myDataset
```
If your load function involves any special arguments, modify the function ```parse_args``` on line [52] accordingly.

Then, to the ```load_data_from_raw``` function on line [63], add your setting:
```bash
...
elif data == 'myDataset':
    docs, responses = load_myDataset(...)
```

We're almost ready to fit HSTMs. It is highly recommended that you review the description of the list of experiment flags and their default values, by running:
```bash
python -m experiment.run_experiment --help
```
For example, take note of flags like ```num_topics, train_test_mode, train_size, C, C_topics```. These tend to be dataset specific. You may need to add new flags to support any special arguments your ```load_myDataset``` function. 

Finally, you can run
```bash
python -m experiment.run_experiment --data=mySetting --model=hstm-all
```

We recommend stepping through the scripts we used to produce tables 1 and 2 (in the previous section) as a template for tuning parameters like ```C``` and ```C_topics```. 


**References**
- [2] Mcauliffe, J. and Blei, D., 2007. _Supervised topic models_. In NeurIPS.
- [3] Card, D., Tan, C., and Smith, N.A., 2017. _Neural Models for Documents with Metadata_. In NAACL.
- [4] Devlin, J., Chang, M.W., Lee, K. and Toutanova, K., 2018. _Bert: Pre-training of deep bidirectional transformers for language understanding_. arXiv preprint arXiv:1810.04805.

