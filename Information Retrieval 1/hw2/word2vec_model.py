import torch
import sys
import pickle as pkl
import argparse
from torch import nn
import data_utils
from torch.autograd import Variable
import torch.nn.functional as F

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.t_vocab_embs = nn.Embedding(vocab_size, embedding_dim, sparse=True) #target embedding matrix for whole vocabulary
        self.c_vocab_embs = nn.Embedding(vocab_size, embedding_dim, sparse=True) #context embedding matrix for whole vocabulary

        nn.init.xavier_uniform_(self.t_vocab_embs.weight)
        nn.init.xavier_uniform_(self.c_vocab_embs.weight)


    def forward(self, t, cp, cn):
        '''
        Takes a list of target id's, a list of corresponding positive context
        id's and a list of negative (noise) samples. These lists are converted
        to long tensors and fed forward through the network, after which a loss
        is computed based on the similarity between the target and context
        embeddings.

        Returns:
            loss: scalar
        '''
        t = Variable(torch.LongTensor(t))
        cp = Variable(torch.LongTensor(cp))
        cn = Variable(torch.LongTensor(cn))

        # Forward for positive samples
        t_batch_embs = self.t_vocab_embs(t) #get target embeddings for batch
        cp_batch_embs = self.c_vocab_embs(cp) #get positive embeddings for batch
        loss_p = torch.mul(t_batch_embs, cp_batch_embs).squeeze()
        loss_p = torch.sum(loss_p, dim=1)
        loss_p = F.logsigmoid(-loss_p).squeeze()

        # Forward for negative samples
        cn_batch_embs = self.c_vocab_embs(cn)  #get negative embeddings for batch
        loss_n = torch.bmm(cn_batch_embs, t_batch_embs.unsqueeze(2)).squeeze()
        loss_n = torch.sum(loss_n, dim=1)
        loss_n = F.logsigmoid(+loss_n).squeeze()

        loss = sum(loss_p+loss_n)/len(t)
        return(loss)
