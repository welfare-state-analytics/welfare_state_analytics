from pytorch_transformers import BertTokenizer
from fastai.text.transform import Tokenizer,BaseTokenizer,List,Vocab
from fastai.text import Learner
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor
from fastai.core import ifnone,is_listy
from fastai.torch_core import Rank0Tensor,OptLossFunc,OptOptimizer,Optional,to_detach
from fastai.callback import CallbackHandler,Tuple,Union

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
         self._pretrained_tokenizer = tokenizer
         self.max_seq_len = max_seq_len
    def __call__(self, *args, **kwargs):
         return self
    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]


def accuracy(y_pred:Tensor, y_true:Tensor)->Rank0Tensor:
    return (y_true.argmax(axis=1) == y_pred.argmax(axis=1)).float().mean()


def loss_batch_bert(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    out = model(*xb)
    out = out[0]
    out = cb_handler.on_loss_begin(out)

    if not loss_func: return to_detach(out), yb[0].detach()
    loss = loss_func(out, *yb)

    if opt is not None:
        loss,skip_bwd = cb_handler.on_backward_begin(loss)
        if not skip_bwd:                     loss.backward()
        if not cb_handler.on_backward_end(): opt.step()
        if not cb_handler.on_step_end():     opt.zero_grad()

    return loss.detach().cpu()


def loss_batch_bert(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    out = model(*xb)
    out = out[0]
    out = cb_handler.on_loss_begin(out)

    if not loss_func: return to_detach(out), yb[0].detach()
    loss = loss_func(out, *yb)

    if opt is not None:
        loss,skip_bwd = cb_handler.on_backward_begin(loss)
        if not skip_bwd:                     loss.backward()
        if not cb_handler.on_backward_end(): opt.step()
        if not cb_handler.on_step_end():     opt.zero_grad()

    return loss.detach().cpu()


def fill(data, n):
    a = int(n/(len(data)-1)+1)
    ret = pd.concat(a * [ data ], ignore_index=True)[0:n]
    
    return ret


def fill_categories(df, label_cols, n=None, random_state=None):
    label_start_index = list(df.columns).index(label_cols[0])
    data_sorted = df.sort_values(by=list(label_cols), ascending=False)
    sums = [ sum(data_sorted[x]) for x in data_sorted.columns[label_start_index:] ]
    index = [ sum(sums[:i]) for i,_ in enumerate(sums) ] + [ len(data_sorted) ]
    m = max(sums)

    return pd.concat([ fill(data_sorted[index[i]:index[i+1]], n=n or m) for i in range(len(sums)) ], ignore_index=True).sample(frac=1, random_state=random_state)
