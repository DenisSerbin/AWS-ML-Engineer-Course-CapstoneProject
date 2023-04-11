import csv
import time
import math
import os
import pandas as pd
import numpy as np
import boto3
import json
import copy
import logging
import sys
import random
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
import sentence_transformers
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# Configuration for Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class CFG:
    def __init__(self, model_name, training_dir, epochs, batch_size):
        self.print_freq = 500
        self.num_workers = 4
        self.model = model_name
        self.training_dir = training_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gradient_checkpointing = False
        self.num_cycles = 0.5
        self.warmup_ratio = 0.1
        self.epochs = epochs
        self.encoder_lr = 1e-5
        self.decoder_lr = 1e-4
        self.eps = 1e-6
        self.betas = (0.9, 0.999)
        self.batch_size = batch_size
        self.weight_decay = 0.01
        self.max_grad_norm = 0.012
        self.max_len = 512
        self.seed = 2016


def seed_everything(cfg):
    random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True


def f2_score(y_true, y_pred):
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    y_true = y_true.apply(lambda x: set(x.split(" ")))
    y_pred = y_pred.apply(lambda x: set(x.split(" ")))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    #f2 = (5 * precision * recall) / (4 * precision + recall)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)


def read_data(cfg):
    train = pd.read_csv(os.path.join(cfg.training_dir, 'sup_train.csv'), index_col = 0)
    correlations = pd.read_csv(os.path.join(cfg.training_dir, 'correlations.csv'), index_col = 0)

    train['title1'].fillna("Title does not exist", inplace = True)
    train['title2'].fillna("Title does not exist", inplace = True)
    
    # Create feature column
    train['text'] = train['title1'] + '[SEP]' + train['title2']
    print(' ')
    print('-' * 50)
    print(f"train.shape: {train.shape}")
    print(f"correlations.shape: {correlations.shape}")
    
    return train, correlations


def get_max_length(train, cfg):
    lengths = []
    for text in tqdm(train['text'].fillna("").values, total = len(train)):
        length = len(cfg.tokenizer(text, add_special_tokens = False)['input_ids'])
        lengths.append(length)
    cfg.max_len = max(lengths) + 2 # cls & sep
    print(f"max_len: {cfg.max_len}")

    
def prepare_input(text, cfg):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
        max_length = cfg.max_len,
        pad_to_max_length = True,
        truncation = True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs


class custom_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['text'].values
        self.labels = df['label'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        label = torch.tensor(self.labels[item], dtype = torch.float)
        return inputs, label


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis = 1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs


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
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states = True)
        self.model = AutoModel.from_pretrained(cfg.model, config = self.config)
        self.config.hidden_dropout = 0.1
        self.config.hidden_dropout_prob = 0.1
        self.config.attention_dropout = 0.1
        self.config.attention_probs_dropout_prob = 0.1
        self.pool = MeanPooling()
        #self.fc = nn.Linear(self.config.hidden_size, 1)
        self.fc = nn.Sequential(nn.Linear(self.config.hidden_size, 256),
                                nn.Dropout(p = 0.2),
                                nn.ReLU(inplace = True),
                                nn.Linear(256, 64),
                                nn.Dropout(p = 0.2),
                                nn.ReLU(inplace = True),
                                nn.Linear(64, 16),
                                nn.Dropout(p = 0.2),
                                nn.ReLU(inplace = True),
                                nn.Linear(16, 1)
                               )
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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled = True)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, target) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        with torch.cuda.amp.autocast(enabled = True):
            y_preds = model(inputs)
            loss = criterion(y_preds.view(-1), target)
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1
        scheduler.step()
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.8f}({loss.avg:.8f}) '
                  'Grad: {grad_norm:.8f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch + 1, 
                          step, 
                          len(train_loader), 
                          remain = timeSince(start, float(step + 1) / len(train_loader)),
                          loss = losses,
                          grad_norm = grad_norm,
                          lr = scheduler.get_lr()[0]))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device, cfg):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, target) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1), target)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.8f}({loss.avg:.8f}) '
                  .format(step, 
                          len(valid_loader),
                          loss = losses,
                          remain = timeSince(start, float(step + 1) / len(valid_loader))))
    predictions = np.concatenate(preds, axis = 0)
    return losses.avg, predictions


def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0
    best_threshold = 0.01
    for thres in np.arange(0.01, 1, 0.01):
        x_val['predictions'] = np.where(val_predictions > thres, 1, 0)
        x_val1 = x_val[x_val['predictions'] == 1]
        x_val1 = x_val1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
        x_val1['content_ids'] = x_val1['content_ids'].apply(lambda x: ' '.join(x))
        x_val1.columns = ['topic_id', 'predictions']
        x_val0 = pd.Series(x_val['topics_ids'].unique())
        x_val0 = x_val0[~x_val0.isin(x_val1['topic_id'])]
        x_val0 = pd.DataFrame({'topic_id': x_val0.values, 'predictions': ""})
        x_val_r = pd.concat([x_val1, x_val0], axis = 0, ignore_index = True)
        x_val_r = x_val_r.merge(correlations, how = 'left', on = 'topic_id')
        x_val_r['content_ids'].fillna(" ", inplace = True)
        x_val_r['predictions'].fillna(" ", inplace = True)
        score = f2_score(x_val_r['content_ids'], x_val_r['predictions'])
        if score > best_score:
            best_score = score
            best_threshold = thres
    return best_score, best_threshold


def train_and_evaluate(train, correlations, cfg):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get model
    model = sup_model(cfg)
    model.to(device)
    
    # Split train & validation
    x_train, x_val = train_test_split(train, test_size = 0.1)
        
    valid_labels = x_val['label'].values
        
    train_dataset = custom_dataset(x_train, cfg)
    valid_dataset = custom_dataset(x_val, cfg)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size = cfg.batch_size,
        shuffle = True,
        num_workers = cfg.num_workers,
        pin_memory = True,
        drop_last = True
    )
        
    valid_loader = DataLoader(
        valid_dataset,
        batch_size = cfg.batch_size,
        shuffle = False,
        num_workers = cfg.num_workers,
        pin_memory = True,
        drop_last = False
    )
    
    # Optimizer
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay = 0.0):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
            'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters
    
    optimizer_parameters = get_optimizer_params(
        model, 
        encoder_lr = cfg.encoder_lr, 
        decoder_lr = cfg.decoder_lr,
        weight_decay = cfg.weight_decay
    )
    
    optimizer = AdamW(
        optimizer_parameters, 
        lr = cfg.encoder_lr, 
        eps = cfg.eps, 
        betas = cfg.betas
    )
    
    num_train_steps = int(len(train) * 0.9 / cfg.batch_size * cfg.epochs)
    num_warmup_steps = num_train_steps * cfg.warmup_ratio
    
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps, 
        num_training_steps = num_train_steps, 
        num_cycles = cfg.num_cycles
        )
    
    # Criterion
    criterion = nn.BCEWithLogitsLoss(reduction = "mean")
    
    # Training & Validation loop
    best_score = 0
    best_threshold = None
    val_predictions = None
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(cfg.epochs):
        start_time = time.time()
        
        # Train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg)
        
        # Validation
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device, cfg)
        
        # Compute f2_score
        score, threshold = get_best_threshold(x_val, predictions, correlations)
        elapsed = time.time() - start_time
        
        print(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.8f}  avg_val_loss: {avg_val_loss:.8f}  time: {elapsed:.0f}s')
        print(f'Epoch {epoch + 1} - Score: {score:.8f} - Threshold: {threshold:.8f}')
        
        if score > best_score:
            best_score = score
            val_predictions = predictions
            print(f'Epoch {epoch + 1} - Save Best Score: {best_score:.8f} Model')
            best_model_wts = copy.deepcopy(model.state_dict())
            
    torch.cuda.empty_cache()
    print('Training complete')
    
    # Get best threshold
    best_score, best_threshold = get_best_threshold(x_val, val_predictions, correlations)
    print(f'The CV score is {best_score:.8f} using a threshold of {best_threshold:.8f}')
    
    model.load_state_dict(best_model_wts)
    return model


def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    
    logger.info(f"Hyperparameters: Batch Size: {args.batch_size}, number of epochs: {args.epochs}")
    
    cfg = CFG(args.model_name, args.training_dir, args.epochs, args.batch_size)
    seed_everything(cfg)

    train, correlations = read_data(cfg)
    
    get_max_length(train, cfg)
    
    # Train and evaluate one fold
    model = train_and_evaluate(train, correlations, cfg)
    
    save_path = args.model_dir + "/model.pth"
    logger.info(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type = int, default = 1)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--model_name", type = str)
    
    # Data, model, and output directories
    parser.add_argument("--model_dir", type = str, default = os.environ["SM_MODEL_DIR"])
    parser.add_argument('--output_dir', type = str, default = os.environ['SM_OUTPUT_DIR'])
    parser.add_argument("--training_dir", type = str, default = os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()

    main(args)