# Capstone Project "Information retrieval using a retriever-reranker ensemble in AWS SageMaker"

The project is motivated by the Kaggle competition “Learning Equality - Curriculum Recommendations” (https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations), which focuses on building efficient ML models that could match educational content (files and videos in all kinds of formats) to curriculum (K-12) topics.

## Datasets

The original training dataset is given in three files: 
 - 'topics.csv' - curriculum topics with descriptions,
- 'content.csv' - content items with descriptions,
- 'correlations.csv' - an alignment of the topics with the content items.

The files are too large to be uploaded here. They can be downloaded from the competition page https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/data

Generated datasets:
- [uns_train.csv](GeneratedData/uns_train.csv) - The training data for the retriever model: it consists of a list of *(topic title, content item title)* pairs known to be related.
- [sup_train.csv](GeneratedData/sup_train.csv) - The training data for the reranker model: it consists of a list of *(topic title, content item title)*, where *topic* ranges through all given topics and *content item* is in the list of content items output by the retriever. The pair *(topic title, content item title)* is labeled by 1 if the *topic* and *content item* are related (the pair is listed in the 'correlations.csv' file), or by 0 is the *topic* and *content item* are not related.
- [sup_test.csv](GeneratedData/sup_test.csv) - The test data for the reranker model.

## Project structure
- [Retriever-Reranker.ipynb](Retriever-Reranker.ipynb) - The main file (jupyter notebook) that calls training and inference scripts for both models, and outputs test results.
- [uns_train.py](Scripts/uns_train.py) - The Python script that defines and trains the retriever model.
- [uns_inference.py](Scripts/uns_inference.py) - The Python script used by the deployed endpoint of the retriever to create embeddings of topics and content items into vector space.
- [sup_train.py](Scripts/sup_train.py) - The Python script that defines and trains the reranker model.
- [sup_inference.py](Scripts/sup_inference.py) - The Python script used by the deployed endpoint of the reranker to predict connections between topics and content items.
- [requirements.txt](Scripts/requirements.txt) - The text file listing packages necessary for the above scripts.

## Retriever

A retriever is a model that for a given topic outputs *a large number* of content items, many of which are not really relevant, but we keep them for the re-ranking stage.

### Fine-tuning of a pretrained sentence transformer

First we fine-tune a pretrained model **'paraphrase-distilroberta-base-v2'** from **'HuggingFace Library'** in an unsupervised fashion, on a set of *positive* *(topic title, content item title)* pairs ([uns_train.csv](GeneratedData/uns_train.csv)), that is, the corresponding topic and content item are known to be related.

### Embedding

Using the fine-tuned model, we map all topic and content item titles to 768-dimensional real-valued vectors and split content title vectors into clusters of 30 nearest neighbors using the KNN algorithm.

### Retrieval of relevant content

For every topic we compose a list of its content item neighbors (with respect to the constructed embedding into vector space). Then we split the list of topics into a training set ([train_topics.csv](GeneratedData/train_topics.csv)) and a test set ([test_topics.csv](GeneratedData/test_topics.csv)).

### Training set for the reranker

For every topic in the [train_topics.csv](GeneratedData/train_topics.csv) list, we label its neighboring content items with either 0, or 1, based on the known correlation with the topic. The result is a training dataset ([sup_train.csv](GeneratedData/sup_train.csv)) for the reranker model - this is the output of the retriever.

### Testing set for the reranker

Similarly we create a testing dataset ([sup_test.csv](GeneratedData/sup_test.csv)) for the reranker model based on the [test_topics.csv](GeneratedData/test_topics.csv) list.

## Reranker

A reranker is a model that filters the output of the retriver. For every pair *(topic, content item)*, the reranker predicts if the corresponding *topic title* and *content item title* are related (outputs 1), or not related (outputs 0).

### Classifier

We construct a custom classification (0 or 1) model based on **'paraphrase-multilingual-mpnet-base-v2'** from **'HuggingFace Library'** and train it on the [sup_train.csv](GeneratedData/sup_train.csv) dataset.

### Trimming of the retriever output

Using the trained reranker model we drop irrelevant content items that were originally output by the retriver.

## Testing

We test the reranker model on the dataset [sup_test.csv](GeneratedData/sup_test.csv) usng the F2 metric.
