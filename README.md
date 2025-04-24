## Overview

This repository contains a project for sentiment analysis of Urdu text using a dataset of Urdu tweets. The project employs various Natural Language Processing (NLP) techniques to preprocess the data and uses machine learning models for sentiment classification.

## Dataset Information

- **Dataset Source**: The Urdu tweets dataset is used for sentiment analysis.
- **Dataset Link**: [Mendeley Dataset](https://data.mendeley.com/datasets/rz3xg97rm5/1)
- **Additional Resources**:
  - **Urdu Stopwords**: [Kaggle - Urdu Stopwords List](https://www.kaggle.com/datasets/rtatman/urdu-stopwords-list?select=stopwords-ur.txt)
  - **Urdu Affix Lists**: [SourceForge - Urdu Affix Lists](https://sourceforge.net/projects/resource-for-urdu-stemmer/files/Urdu%20Affix%20lists.pdf/download)

## Project Structure

1. **Data Preprocessing**:
   - The dataset is readable and unnecessary columns are removed.
   - **Stop Words Removal**: Urdu stop words are removed using a predefined list.
   - **Tokenization**: The text is tokenized into words.
   - **TF-IDF Vectorization**: TF-IDF vectorizer is used to convert text data into numerical format.
   - **Word2Vec Embedding**: Word2Vec model is used for word embeddings.
   - **N-Grams**: N-grams are generated for the text data.

2. **Model Training and Evaluation**:
   - The dataset is split into training and testing sets.
   - A **Naive Bayes classifier** is used for sentiment classification.
   - **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-Score are calculated to evaluate the model's performance.

## Usage

1. **Dependencies**:
   - Install the required Python libraries/packages:
     ```bash
     pip install pandas nltk scikit-learn gensim numpy
     ```

2. **Running the Notebook**:
   - Open the `Sentiment Analysis.ipynb` Jupyter Notebook.
   - Ensure the dataset is placed correctly and update the path if necessary.
   - Run all the cells sequentially to perform data preprocessing, train the model, and evaluate its performance.

## Features

- **Sentiment Analysis**: Classifies Urdu text into positive, negative, or neutral sentiments.
- **NLP Techniques**: Implements various NLP techniques such as tokenization, stop words removal, TF-IDF vectorization, and word embeddings.
- **Machine Learning Model**: Uses a Naive Bayes classifier for sentiment prediction.

## Results

The model's performance is evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**


