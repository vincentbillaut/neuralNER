# Exploring Neural Architectures for NER

- Vincent Billaut   MS in Statistics, Stanford University
- Marc Thibault     MS in ICME, Stanford University

## Shell instructions to run on GPU

SSH in
```bash
ssh [username]@icme-gpu.stanford.edu
...
```

Screen in
```bash
screen -R neuralner
```

Ask for resources
```bash
srun  --partition=k80 --gres=gpu:1 --time=<minutes> --pty bash
```
for < minutes >, easy to go up to 270, then we get a hold.


Load components
```bash
module purge

module load cuda90/toolkit/9.0.176

module load cudnn/7.0
```


Run the scripts
```zsh
python3.6 main_learning.py train
```

## Project overview

### Problem description

Our goal is to tackle the problem of **Named Entity Recognition** (NER) using neural architectures. More precisely, we ought to reproduce, and try to carefully study the protocol described in [1]. The added value we wish to bring notably consists in a detailed study of **classification errors**, in order to better understand where this approach *fails*, and where/how it can be *improved*.

### Data

We will use the same dataset as the one used in [1], namely **CoNLL-2002** (Tjong Kim Sang, 2002), which we have acquired already, and potentially CoNLL-2003 ([4]) later on, if this proves relevant.

The dataset consists in **1.05M** labeled words (**150MB** csv file).

### Methodology / Algorithm

The classical 2000s approach to NER would rely on minimum entropy models, HMM and SVM (as explained in [4]). These techniques heavily relied on hand-made feature engineering, and therefore didn't prove robust/adaptive enough, especially when we try to generalize them across languages.

Today, the approach has shifted to Neural Nets and feature learning. The basic approaches are

- **CNN** ([2])
- RNN / **LSTM** ([1], [3])

, potentially enriched by conditional random fields.

We will implement and compare these different methods in order to try and reach conclusions on their respective relevance on given specific cases, which we will compare to results from [1].

### Evaluation Plan

We will use the quantitative evaluation metric formalized in [4]: the $\textbf{F}_1$**-score** on the classification task. Furthermore, we will manually examine a subset of the misclassified instances in order to assess the relative relevance of the various techniques on distinct use cases.

## Code overview

Our code -- or at least the first iterations of it -- is very largely inspired from **Stanford's CS224N starter code for [Assignment 2](http://web.stanford.edu/class/cs224n/assignment2/index.html) and [Assignment 3](http://web.stanford.edu/class/cs224n/assignment3/index.html)**. The framework this repo provided seemed to us like a good base to build upon.  
It basically features a set of objects that make the development of a clean `tensorflow` work environment easier.
We made many adjustments to fit our own specifications but tried to be consistent with the way it was coded in the beginning.


## Related Work

> **[1]** Lample, Guillaume, et al. "Neural architectures for named entity recognition." *arXiv preprint arXiv:1603.01360* (2016).

​	This article is the basis that will guide our work. It uses LSTM-CRF models and compares them to transition-based approaches.

> **[2]** Collobert, Ronan, et al. "Natural language processing (almost) from scratch." *Journal of Machine Learning Research* 12.Aug (2011): 2493-2537.

​	This article implements CNN methods for NER.

> **[3]** Huang, Zhiheng, Wei Xu, and Kai Yu. "Bidirectional LSTM-CRF models for sequence tagging." *arXiv preprint arXiv:1508.01991* (2015).

​	This article implements LSTM models very similar to the first approach described in [1], and will prove useful in implementing the neural architecture ourselves. It notably shows how to combine LSTM with CRF.

> **[4]** Tjong Kim Sang, Erik F., and Fien De Meulder. "Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition." *Proceedings of the seventh conference on Natural language learning at HLT-NAACL 2003-Volume 4*. Association for Computational Linguistics, 2003.

​	This article describes very generally the task of named entity recognition, and provides a common dataset which has been used as a standard for NER ever since, and notably by the aforementioned articles.
