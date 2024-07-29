from typing import List
import numpy as np
from gensim import corpora
import torch.nn as nn
import torch
import torch.optim as optim
from core.model.Classifier import Classifyer
from torch.nn.utils.rnn import pad_sequence
from core.model.backbone.cnn1d import CNN1dEncoder

class SumEmbeddings(nn.Module):
    def __init__(self, vocab_size, dim):
        super(SumEmbeddings, self).__init__()
        self.embedding_layer = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=dim
            )

    def forward(self, input):
        embedded_input = self.embedding_layer(input)
        summed_embedding = torch.sum(embedded_input, dim=1)
        
        return summed_embedding

class CNNW2VEncoder:
    def __init__(self, seq_hidden, embedding_dim=100, epochs=50):
        self.dim = embedding_dim
        self.epochs = epochs
        self.seq_hidden = seq_hidden

    def fit(self, data_set: List[List[str]], labels):
        self.dictionary = corpora.Dictionary(data_set)
        self.word2idx=self.dictionary.token2id
        
        for i, doc in enumerate(data_set):
            data_set[i] = torch.tensor([self.word2idx[w] for w in doc])
        input=pad_sequence(data_set, batch_first=True)
        input = input.long()

        label_vocab = list(set(labels))
        target = torch.LongTensor([label_vocab.index(w) for w in labels])

        self.sequential_encoder = nn.Sequential(
            SumEmbeddings(
                vocab_size=self.dictionary.num_pos,
                dim=self.dim
            ),
            CNN1dEncoder(
                in_dim=1,
                hidden_dim=self.seq_hidden,
                kernel_size=5,
                dropout=0.2
            ),
            Classifyer(in_dim=self.dim,
                    hiddens=[128, 64],
                    out_dim=len(label_vocab))
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.sequential_encoder.parameters(), lr=0.01)
        # train
        for epoch in range(self.epochs):
            self.sequential_encoder.train()
            # word2idx
            output=self.sequential_encoder(input)
            loss=criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"[epoch {epoch}] 1d CNN Loss: {loss.detach().item()}")


        

    def get_sentence_embedding(self, text: List[str]) -> List[float]:
        text = ' '.join(text)
        
        # senetence embedding
        sen_emb = np.array([0] * self.dim, 'float32')
        if text != '':
            words = list(set(text.split(' ')))
            for word in words:
                if word in self.word2idx.keys():
                    word_idx=torch.LongTensor([self.word2idx[word]]).reshape(1,-1)
                    # get the output from CNN layer
                    sen_emb = sen_emb + \
                        self.sequential_encoder[:2](word_idx).flatten().detach().numpy()
                else:
                    continue

        return sen_emb