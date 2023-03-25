import csv
import time
import math
import os
import pandas as pd
import numpy as np
import boto3
import random
import json
import logging
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
import sentence_transformers
from tqdm.auto import tqdm
from io import StringIO

# Configuration for Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class CFG:
    def __init__(self, model_dir):
        self.print_freq = 3000
        self.num_workers = 4
        self.model = model_dir
        self.model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        #self.model_name = 's3://sagemaker-us-east-1-852055550328/huggingface-pytorch-training-2023-03-20-01-39-03-241/output/model.tar.gz'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.gradient_checkpointing = False
        self.batch_size = 32
        self.top_n = 30
        self.seed = 2016
        self.threshold = 0.06

        
def seed_everything(cfg):
    random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    

class sup_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model_name, output_hidden_states = True)
        self.model = AutoModel.from_pretrained(cfg.model_name, config = self.config)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    
    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


def model_fn(model_dir):
    
    cfg = CFG(model_dir)
    seed_everything(cfg)
    
    logger.info("In model_fn. Model directory is -", model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sup_model(cfg)
    logger.info("Model object createdsuccessfully")
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        logger.info("Loading the re-ranker model.")
        logger.info("Model_dir content: ")
        logger.info(os.listdir(model_dir))
        checkpoint = torch.load(f, map_location = device)
        model.load_state_dict(checkpoint)
        #model.load_state_dict(torch.load(f, map_location = device))
        logger.info('Model loaded successfully')
    
    model.eval()
    model.to(device)
    
    return model, cfg.tokenizer


#def input_fn(request_body, request_content_type):
    
#    if content_type == 'text/csv':
        # Read the raw input data as CSV.
#        df = pd.read_csv(StringIO(input_data), header = None)
#        return df
#    else:
#        raise ValueError("{} not supported by script!".format(content_type))


def input_fn(request_body, content_type):
    '''Loading Data from User Input and serialize into list'''

    logger.info("Deserializing the input data")
    if content_type == "application/json":
        data = pd.read_json(request_body, orient = "records", lines = True)

    elif content_type == "text/plain":
        data = pd.DataFrame({'text' : json.loads(request_body)})

    elif content_type == "text/csv":
        data = pd.read_csv(StringIO(request_body))
    else:
        raise Exception(f"Requested unsupported ContentType in content_type: {content_type}")
        
    return data


def prepare_sup_input(text, cfg):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs


class sup_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['text'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_sup_input(self.texts[item], self.cfg)
        return inputs
    

#class sup_dataset(Dataset):
#    def __init__(self, data, cfg):
#        self.cfg = cfg
#        self.texts = data
#    def __len__(self):
#        return len(self.texts)
#    def __getitem__(self, item):
#        inputs = prepare_sup_input(self.texts[item], self.cfg)
#        return inputs


def inference_fn(test_loader, model, device):
    preds = []
    for inputs in tqdm(test_loader, total = len(test_loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
    predictions = np.concatenate(preds)
    return predictions


def predict_fn(data, model_and_tokenizer):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer
    cfg = model.cfg
        
    test_dataset = sup_dataset(data, cfg)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers, 
        pin_memory = True, 
        drop_last = False
    )
    
    predictions = inference_fn(test_loader, model, device)
    
    return predictions