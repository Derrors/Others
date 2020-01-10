# -*- coding: utf-8 -*-
# author: Qinghe Li
# date: 2019/10/10

import argparse
import random
import logging
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from apex.optimizers import FP16_Optimizer, FusedAdam
from tqdm import trange
import numpy as np

from transformers.optimization import AdamW
from transformers.modeling_bert import BertForTokenClassification

from data_loader import DataLoader
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='./bert-base-chinese', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--model_dir', default='./model', help="Directory containing params.json")
parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir containing weights to reload before training")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")


def train(model, data_iterator, optimizer, scheduler, params):
    # set model to training mode
    model.train()
    optimizer.step()
    scheduler.step()

    # a running average object for loss
    loss_avg = utils.RunningAverage()
    
    # Use tqdm for progress bar
    t = trange(params.train_steps)
    for i in t:
        # fetch the next training batch
        batch_data, batch_labels = next(data_iterator)
        batch_masks = batch_data.gt(0)

        # compute model output and loss
        outputs = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs[0]

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))


def evaluate(model, data_iterator, params, mark='Eval'):
    # set model to evaluation mode
    model.eval()

    idx2label = params.idx2label

    true_labels = []
    pred_labels = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    for _ in range(params.eval_steps):
        # fetch the next evaluation batch
        batch_data, batch_labels = next(data_iterator)
        batch_masks = batch_data.gt(0)

        outputs_train = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs_train[0]
        
        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()
        loss_avg.update(loss.item())
        
        outputs_test = model(batch_data, token_type_ids=None, attention_mask=batch_masks)  # shape: (batch_size, max_len, num_labels)
        batch_output = outputs_test[0]
        batch_output = batch_output.detach().cpu().numpy()
        batch_labels = batch_labels.to('cpu').numpy()

        pred_labels.extend([idx2label.get(idx) for indices in np.argmax(batch_output, axis=2) for idx in indices])
        true_labels.extend([idx2label.get(idx) for indices in batch_labels for idx in indices])
    assert len(pred_labels) == len(true_labels)
    
    pred = np.array(pred_labels)
    true = np.array(true_labels)
    num_correct = np.sum(pred==true, dtype=int)
    num_pred = len(pred_labels)
    num_true = len(true_labels)

    precision = 100 * num_correct / num_pred
    recall = 100 * num_correct / num_true
    f1 = 2 * precision * recall / (precision + recall)

    # logging loss, f1 and report
    metrics = {}
    f1 = f1_score(true_labels, pred_labels)
    metrics['Loss'] = loss_avg()
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1'] = f1
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    return metrics


def train_and_evaluate(model, train_data, test_data, optimizer, scheduler, params, model_dir, restore_file=None):
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
        
    best_test_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Compute number of batches in one epoch
        params.train_steps = params.train_size // params.batch_size
        params.test_steps = params.test_size // params.batch_size

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        # Train for one epoch on training set
        train(model, train_data_iterator, optimizer, scheduler, params)

        # data iterator for evaluation
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=False)
        test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)

        # Evaluate for one epoch on training set and validation set
        params.eval_steps = params.train_steps
        train_metrics = evaluate(model, train_data_iterator, params, mark='Train')
        params.eval_steps = params.test_steps
        test_metrics = evaluate(model, test_data_iterator, params, mark='Test')
        
        test_f1 = test_metrics['f1']
        improve_f1 = test_f1 - best_test_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer.optimizer if args.fp16 else optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_to_save.state_dict(),
                               'optim_dict': optimizer_to_save.state_dict()},
                               is_best=improve_f1>0,
                               checkpoint=model_dir)
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_test_f1 = test_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("Best test F1: {:05.2f}".format(best_test_f1))
            break
        

if __name__ == '__main__':
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()
    print(params.n_gpu)
    params.multi_gpu = args.multi_gpu

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("device: {}, n_gpu: {}, 16-bits training: {}".format(params.device, params.n_gpu, args.fp16))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # Initialize the DataLoader
    data_loader = DataLoader(args.data_dir, args.bert_model_dir, params, token_pad_idx=0)
    
    # Load training data and test data
    train_data = data_loader.load_data('train')
    test_data = data_loader.load_data('test')

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.test_size = test_data['size']

    # Prepare model
    model = BertForTokenClassification.from_pretrained(args.bert_model_dir, num_labels=len(params.label2idx))
    model.to(params.device)
    if args.fp16:
        model.half()

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if params.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    if args.fp16:
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=params.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=params.learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model, train_data, test_data, optimizer, scheduler, params, args.model_dir, args.restore_file)
