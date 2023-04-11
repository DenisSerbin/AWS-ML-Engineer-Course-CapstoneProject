import csv
import os
import pandas as pd
import boto3
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses, InputExample
from datasets import Dataset
import random
import logging
import sys
import argparse
import torch


def create_train_sentences(dataset):
    train_examples = []
    train_data = dataset["set"]
    n_examples = dataset.num_rows

    for i in range(n_examples):
        example = train_data[i]
        train_examples.append(InputExample(texts = [str(example[0]), str(example[1])], label = random.uniform(0.6, 0.9)))
        #train_examples.append(InputExample(texts = [str(example[0]), str(example[1])]))
    
    return train_examples

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type = int, default = 2)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--warmup_steps", type = int, default = 100)
    parser.add_argument("--model_name", type = str)
    #parser.add_argument("--learning_rate", type = str, default = 5e-5)
    
    # Data, model, and output directories
    parser.add_argument("--model_dir", type = str, default = os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training_dir", type = str, default = os.environ["SM_CHANNEL_TRAIN"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level = logging.getLevelName("INFO"),
        handlers = [logging.StreamHandler(sys.stdout)],
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    #word_embedding_model = models.Transformer(args.model_name)
    #pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
    #                               pooling_mode_mean_tokens = False, 
    #                               pooling_mode_cls_token = True)
    #model = SentenceTransformer(modules = [word_embedding_model, pooling_model])
    model = SentenceTransformer(args.model_name)
    
    # read data
    print('******* list files *********: ', os.listdir(args.training_dir))
    print('********************** Reading Data *************************')
    train_data = pd.read_csv(os.path.join(args.training_dir, 'uns_train.csv'), index_col = 0)
    
    train_dataset = Dataset.from_pandas(train_data)
    train_examples = create_train_sentences(train_dataset)
    train_dataloader = DataLoader(train_examples, batch_size = args.batch_size, shuffle = True)
    
    # Since we have only positive pairs, use 'MultipleNegativesRankingLoss'
    #train_loss = losses.MultipleNegativesRankingLoss(model = model)
    train_loss = losses.CosineSimilarityLoss(model = model)
    
    #call the fit method
    model.fit(train_objectives = [(train_dataloader, train_loss)],
              epochs = args.epochs,
              warmup_steps = args.warmup_steps,
              #weight_decay = 0,
              #scheduler = 'constantlr',
              #optimizer_params = {'lr': args.learning_rate},
              show_progress_bar = True
             )
    
    model.save(args.model_dir)