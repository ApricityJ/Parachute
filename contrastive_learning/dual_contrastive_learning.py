import math
import copy
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import RobertaTokenizer, RobertaModel, BertModel, BertForMaskedLM, BertConfig, RobertaForMaskedLM



MAX_LENGTH = 512


class CrossEntropy(nn.Module):
    ''' Cross-entropy loss with syntactic regularization term and label-smoothing '''

    def __init__(self, opt):
        super(CrossEntropy, self).__init__()
        self.confidence = 1.0 - opt.eps
        self.smoothing = opt.eps
        self.classes = opt.label_class
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.opt = opt
        self.gamma = opt.gamma
        self.keep_ratio = 0.1
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, labels,
                keep_prob=None, wordpiece_mask=None,
                if_mixup=False):
        # outputs: (bs, nc), labels: (bs, ), drop_prob (bs, sl, 1), wordpiece_mask (bs, sl)
        predicts, _, _ = outputs  # predicts: (bs, 2)
        predicts = self.logsoftmax(predicts)
        with torch.no_grad():
            true_dist = torch.zeros_like(predicts)
            true_dist.fill_(self.smoothing / (self.classes - 1))  # tensor全部fill某个值
            if if_mixup:
                true_dist = labels  # labels变成true_dist的转换需要在函数外进行
            else:
                true_dist.scatter_(1, labels.data.unsqueeze(1), self.confidence)  # ().scatter_(dim, index, src)

        loss_all = -torch.mean(torch.sum(true_dist * predicts, dim=-1))
        if keep_prob is not None:
            text_len = torch.sum(wordpiece_mask, dim=1)  # (bs, )
            keep_num = torch.multiply(text_len, self.keep_ratio)  # (bs, )
            if self.opt.loss_mode == "baseline":
                pass
            elif self.opt.loss_mode == "la":
                loss_all += self.gamma * self.mse_loss(torch.sum(keep_prob, dim=1).squeeze(-1), keep_num)  # 注意力分配
            elif self.opt.loss_mode == "l1":
                loss_all += self.gamma * torch.mean((torch.sum(torch.abs(keep_prob.squeeze(-1)), dim=1) / text_len))  # 注意力L1 loss
            elif self.opt.loss_mode == "lal1":
                loss_all += self.gamma * self.mse_loss(torch.sum(keep_prob, dim=1).squeeze(-1), keep_num)  # 注意力分配
                loss_all += self.gamma * torch.mean((torch.sum(torch.abs(keep_prob.squeeze(-1)), dim=1) / text_len))  # 注意力L1 loss
            else:
                raise ValueError
        return loss_all

    def kl_divergence(self, p_logit, q_logit):
        # p_logit: [batch, hd]
        # q_logit: [batch, hd]
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
        return torch.mean(_kl)

class DynamicLSTM(nn.Module):
    '''
    LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, lenght...).
    '''

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        '''
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort
        '''
        total_length = x.size(1) if self.batch_first else x.size(0)
        '''sort'''
        x_sort_idx = torch.sort(x_len, descending=True).indices
        x_unsort_idx = torch.sort(x_sort_idx).indices
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        '''pack'''
        # print(x_len.min(), x_len.max())
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(), batch_first=self.batch_first)
        ''' process '''
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        '''unsort'''
        ht = ht[:, x_unsort_idx]
        if self.only_use_last_hidden_state:
            return ht
        else:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first,
                                                            total_length=total_length)
            if self.batch_first:
                out = out[x_unsort_idx]
            else:
                out = out[:, x_unsort_idx]
            if self.rnn_type == 'LSTM':
                ct = ct[:, x_unsort_idx]
            return out, (ht, ct)


class AttBERTForPolarity(nn.Module):
    def __init__(self, opt):
        super(AttBERTForPolarity, self).__init__()
        WD = opt.word_dim  # dimension of word embeddings, here 768
        LC = opt.label_class
        self.opt = opt

        bert = BertModel.from_pretrained('bert-base-uncased')
        for param in bert.parameters():
            param.requires_grad = True
        ''' fc_out '''
        self.fc_dropout = nn.Dropout(opt.fc_dropout)
        ''' bert lm model '''
        self.bert_lm_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(opt.device)
        self.bert_lm_model.bert = bert  # 共享地址
        ''' label feature '''
        self.label_trans = nn.Linear(WD, WD)
        self.label_activation = nn.Tanh()
        self.label_dropout = nn.Dropout(opt.fc_dropout)
        ''' cls feature '''
        self.cls_trans = nn.Linear(WD, WD)
        self.cls_activation = nn.Tanh()
        self.cls_dropout = nn.Dropout(opt.fc_dropout)

    def forward_language_model(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        return self.bert_lm_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                  attention_mask=attention_mask, labels=labels)

    def resize_vocab(self, bert_tokenizer):
        self.bert_lm_model.bert.resize_token_embeddings(len(bert_tokenizer))
        self.bert_lm_model.resize_token_embeddings(len(bert_tokenizer))

    def forward(self, inputs, labels=None, output_attentions=None):
        textp, wordpiece_mask = inputs
        outputs = self.bert_lm_model.bert(textp, attention_mask=wordpiece_mask,
                                          output_attentions=output_attentions)  # (bs, sl, 768)
        return outputs


class ROBERTAForPolarity(nn.Module):
    def __init__(self, opt):
        super(ROBERTAForPolarity, self).__init__()
        WD = opt.word_dim  # dimension of word embeddings, here 768
        LC = opt.label_class
        self.opt = opt

        roberta = RobertaModel.from_pretrained('roberta-base')
        for param in roberta.parameters():
            param.requires_grad = True
        ''' fc_out '''
        self.fc_dropout = nn.Dropout(opt.fc_dropout)
        ''' bert lm model '''
        self.roberta_lm_model = RobertaForMaskedLM.from_pretrained('roberta-base').to(opt.device)
        self.roberta_lm_model.roberta = roberta  # 共享地址
        ''' label feature '''
        self.label_trans = nn.Linear(WD, WD)
        self.label_activation = nn.Tanh()
        self.label_dropout = nn.Dropout(opt.fc_dropout)
        ''' cls feature '''
        self.cls_trans = nn.Linear(WD, WD)
        self.cls_activation = nn.Tanh()
        self.cls_dropout = nn.Dropout(opt.fc_dropout)

    def forward_language_model(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        return self.roberta_lm_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                     attention_mask=attention_mask, labels=labels)

    def resize_vocab(self, roberta_tokenizer):
        self.roberta_lm_model.roberta.resize_token_embeddings(len(roberta_tokenizer))
        self.roberta_lm_model.resize_token_embeddings(len(roberta_tokenizer))

    def forward(self, inputs, labels=None, output_attentions=None):
        textp, wordpiece_mask = inputs
        outputs = self.roberta_lm_model.roberta(textp, attention_mask=wordpiece_mask,
                                                output_attentions=output_attentions)  # (bs, sl, 768)
        return outputs