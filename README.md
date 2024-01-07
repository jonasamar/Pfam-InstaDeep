# Pfam classification exercise

## Overview

This repository contains the code and notebooks for building a classification model on the Pfam database using various methods. The project explores different approaches, including data exploration, main statistics analysis, and machine learning pipelines.

## Folders

- **DeepLibrary:**
  - Hand-written package with utility functions for the BPEncoding and Bert model pipeline (Notebooks/BPE_Bert_Pipeline.ipynb).

- **Notebooks:**
  - `Data_Exploration_Statistics.ipynb`: Notebook for data exploration and main statistics analysis.
  - `BPE_BERT_Pipeline.ipynb`: Notebook for the BPE and BERT pipeline.
  - `ML_Pipeline_Features_From_Proteins.ipynb`: Notebook for the machine learning pipeline on features extracted from protein sequences.
  - `KNN_Pipeline.ipynb`: Notebook building KNN algorithm to classify proteins.
  - `Kaggle_Notebook.ipynb`: Notebook available on kaggle [https://www.kaggle.com/code/ulricharmel/pfam-classification](URL).

- **random_split:**
  - Folder containing the Pfam database.

## Project Overview

To address the protein classification problem in the pfam database, I began by exploring the data structure. During this exploration, several issues were identified and addressed, including the presence of duplicates in the training, validation, and test tables, as well as sequences appearing in multiple tables simultaneously (potentially impacting the final model performance tests slightly). The amino acid sequences themselves varied widely in length (from 4 to 2037) with a distribution reflecting the presence of 20 main and 5 rarer amino acids. The train, validation, and test splits seemed to have been done in proper proportions, with 80%, 10%, and 10% of the data, and a similar class distribution across all three tables.

The first attempt at building a classification model involved using a BERT model with BPE encoding of the sequences. However, this approach did not yield successful results. After encoding, the need for a more powerful machine or a remote server capable of running for extended periods (Google Colab disconnected after only 2-3 hours each time) led us to abandon this initial path.

Seeking to simplify the problem, I opted for more traditional machine learning models, such as logistic regressions (regularized or not), XGBoost, and RandomForest. To represent the sequences, I decided to use features I devised myself: the frequency of amino acids in the chain and the frequency with which an amino acid follows another (the frequency of AB being the number of iterations of AB divided by the number of iterations of A). The table created for training became quite large, prompting the idea of training models on a subset of the training dataset (5%, 10%, 15%, etc.). However, even for the smallest percentages, the ML algorithms did not stop running.

I then turned to an approach using clustering (KNN) based on a simple distance measure (Hamming distance). Similar to my previous script, I trained KNN on 2%, 5%, 8%, etc. However, the distance matrices, being square matrices, quickly grew too large for my laptop or Google Colab to handle, leading to unsatisfactory results, especially for the 2% subset. Consequently, I abandoned this method.

Given time constraints and the need to present results, I ran an existing notebook on Kaggle. One critical aspect I overlooked in my previous scripts was likely the extraction of the most represented protein classes to create a balanced dataset (which I had not done before). This could have reduced the size of my variables (likely addressing storage issues) and simplified the classification task (from 17,929 classes to 100 classes).

## Results

It appears that machine learning, particularly deep learning, is well-suited for the protein classification problem based on amino acid sequences. The results obtained on the Kaggle notebook are indeed promising, and other notebooks using different methods might have provided complementary and equally interesting results if applied to a problem with a balanced dataset and fewer classes.

Jonas Amar
Email: jonas.amar@etu.minesparis.psl.eu