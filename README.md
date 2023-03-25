# Capstone Project "Information retrieval using a retriever-reranker ensemble in AWS SageMaker"

The project is motivated by the Kaggle competition “Learning Equality - Curriculum Recommendations” (https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations), which focuses on building efficient ML models that could match educational content (files and videos in all kinds of formats) to curriculum (K-12) topics.

## Datasets

The training dataset is given in three files:
 
 - 'topics.csv' - curriculum topics with descriptions,
- 'content.csv' - content items with descriptions,
- 'correlations.csv' - an alignment of the topics with the content items.

All the files can be downloaded from the competition page https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/data

## Retriever part

A retriever is a model that for a given topic outputs *a significant number* of content items, many of which are not really relevant, but we keep them for the re-ranking stage.

### Fine-tuning of a pretrained sentence transformer

First we fine-tune a pretrained model **'paraphrase-distilroberta-base-v2'** from **'HuggingFace Library'** in an unsupervised fashion, on a set of *positive* (topic title, content item title) pairs (`uns_train.csv`), that is, the corresponding topic and content item are known to be related.

### Embedding

Using the fine-tuned model, we map all topic and content item titles to 768-dimensional real-valued vectors and split content title vectors into clusters of 30 nearest neighbors using the KNN algorithm.

### Retrieval of relevant content

For every topic we compose a list of its content item neighbors (with respect to the constructed embedding into vector space). Then we split the list of topics into a training set (`train_topics.csv`) and a test set (`test_topics.csv`).

### Training set for the reranker

For every topic in the `train_topics.csv` list, we label its neighboring content items with either 0, or 1, based on the known correlation with the topic. The result is a training dataset (`sup_train.csv`) for the reranker model - this is the output of the retriever.
