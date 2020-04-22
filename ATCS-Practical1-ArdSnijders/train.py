import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_printoptions(threshold=5000)

import nltk.tokenize 
nltk.download('punkt')
import argparse
import numpy as np
import sys

import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

def build_iters():
    """
    Utility function for loading SNLI data, constructing Field objects/vocabulary, loading vectors,
    Performing split.
    :return: iterables for training loop, data.Field objects for vocab_size
    """
    # Define tokenize function; build TEXT and LABEL field objects.
    print('Building Field objects..',flush=True)
    TEXT = data.Field(lower=True, tokenize=nltk.tokenize.word_tokenize, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

    print('Loading SNLI data, performing data split',flush=True)
    # Create splits for data. returns torchtext.datasets.nli.SNLI objects
    train, val, test = datasets.SNLI.splits(TEXT, LABEL)

    print('Loading GloVe vectors..',flush=True)
    # Build vocabulary with training set, using GloVe vectors
    TEXT.build_vocab(train, vectors='glove.840B.300d')
    LABEL.build_vocab(train)

    print('Building BucketIterators..',flush=True)
    # Builds the BucketIterators for train/val/test -> These are simply Torchtext equivalents of Pytorch Dataloaders.
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=config.batch_size, sort_key=lambda x: len(x.premise))  # , device=0)

    return TEXT, train_iter, val_iter, test_iter

class MLPClassifier(nn.Module):
    """
    Class for Multi-Layer Perceptron Classifier
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Initializes the network
        :param input_dim: dimension of first layer
        :param hidden_dim: dimension of hidden layer
        """
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, input):
        """
        Method to do the forward pass
        :param input: batch of concatenations of [u,v,u-v,u*v]
        :return: [batch_size, 3] tensor with predictions
        """

        output = self.network(input)
        return output

#TODO average embeddings
class AverageEmbeddings(nn.Module):
    """
    Simple network that takes average of all word embeddings in the sentence
    """
    def __init__(self, vocab_size, emb_dim):

        super(AverageEmbeddings, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=emb_dim, mode="mean")
        self.embedding.weight.requires_grad=False

    def forward(self, sentences, offsets):
        """
        the forward pass of the model
        :param sentences: sequence representing hypothesis/premise sentence
        :param offsets: length of offset
        :return:
        """
        offsets = None
        embedded = self.embedding(sentences, offsets)
        return embedded

#TODO: IMPLEMENT LSTM ENCODER CLASS
class LSTM_Encoder(nn.Module):
    """
    Class for all LSTM encoder models
    """
    def __init__(self, vocab_size, hidden_dim, emb_dim, num_layers, bidirectional, maxpool):
        """
        Initializes the LSTM network
        :param vocab_size: size of the vocabulary for initializing nn.Embedding
        :param hidden_dim: dimensionality of hidden layer in LSTM
        :param emb_dim: dimensionality of word embeddings
        :param num_layers: number of "stacked" LSTM layers (always 1)
        :param bidirectional: flag to turn on/off bidirectional encoding
        """
        super(LSTM_Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.maxpool = maxpool

        # Initialize embedding layers. Turn off gradient tracking as we don't want to update the embeddings.
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.requires_grad=False

        # Layers to initialize hidden state & cell state(?)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=config.dropout)

    def forward(self, sentences, offsets):
        """
        Performs the forward pass through the network
        :param sentences: batch of sequences representing hypotheses or premises sentences
        :param offsets: tensor with number of padding tokens per sentence
        :return:
        - if unidirectional: ([batch_size, hidden_dim]) shaped tensor with last hidden state of network
        - if bidirectional: ([batch_size, 2*hidden_dim]) shaped tensor with concatenated forward/backward hidden states
        - if bidirectional and max_pooling: ([batch_size, 2*hidden_dim]) tensor with max-pool on features for all t
        """
        embedded = self.embedding(sentences)

        #this function packs padded variable length sequences into a PackedSequence object, which we can feed into the encoder layer
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, batch_first=True, lengths=offsets, enforce_sorted=False)
        output, (hidden, cell) = self.encoder(packed_embedded)

        # Pad the packed output which contains the hidden states
        # Function returns a tuple with padded outputs and tensor with lengths of each sequence in the batch
        output_padded, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=-9999)

        if self.bidirectional and self.maxpool:
            batch = output_padded
            # Take row-wise maximum per sentence to get max-pooled feature representation of shape ([hidden_dim*2])
            batch = [(torch.max(sent, dim=0, keepdim=True)[0]).squeeze() for sent in batch]
            # Combine list of tensors back into tensor of tensors of shape ([batch_size, hidden_dim*2])
            batch = torch.stack(batch, dim=0)
            result = batch

        # Concatenation of forward and backward LSTM final hidden states
        elif self.bidirectional and not self.maxpool:
            forward_h_s = hidden[0]
            backward_h_s = hidden[1]
            result = torch.cat((forward_h_s, backward_h_s), 1)
        # Uni-directional LSTM
        else:
            result = torch.squeeze(hidden)

        return result

def evaluate(model, classifier, val_iter, criterion):
    """
    Function to perform evaluation on the val and test_iter. Note: this function can also be used to test a pre-trained model.
    :param model: takes in an encoder instance
    :param classifier: an MLP instance
    :param val_iter: takes the validation or test iterable (val_iter / test_iter)
    :param criterion: Torch object to calculate the loss
    :return: lists with validation loss, validation accuracy
    """
    
    if config.mode=="test":
        model = model.to(device)
        classifier = classifier.to(device)

    val_loss = []
    accuracies = []
    # iterate over validation set
    for i, batch in enumerate(val_iter):
        premises, hypotheses = batch.premise, batch.hypothesis
        labels = batch.label - 1  # offset labels by 1 to get indices for criterion
        u = model(sentences=premises[0].to(device), offsets=premises[1].to(device))
        v = model(sentences=hypotheses[0].to(device), offsets=hypotheses[1].to(device))

        # combine vectors u and v into single vector
        differences = torch.abs(u - v)
        products = torch.mul(u, v)
        final_vectors = torch.cat((u, v, differences, products), 1)

        # Pass transformed vectors into classifier, compute loss, store in list
        predictions = classifier(final_vectors)
        loss = criterion(predictions, labels.to(device))
        val_loss.append(loss.item())

        # Compute accuracy on batch, store in list
        _, indices = predictions.max(dim=1)
        accuracy = torch.eq(labels.to(device), indices.to(device)).sum().item() / config.batch_size
        accuracies.append(accuracy)

        if config.print_val:
            if i % (config.print_every/10) == 0 and i > 1:
                print("------- loss on validation set: {} validation accuracy: {}".format(
                    np.asarray(val_loss).mean(),
                    np.asarray(accuracies).mean()     ))
            val_loss = np.asarray(val_loss).mean()
            val_accuracy = np.asarray(accuracies).mean()
            return val_loss, val_accuracy

    # Compute mean loss and mean accuracy on validation set
    val_loss = np.asarray(val_loss).mean()
    val_accuracy = np.asarray(accuracies).mean()
    return val_loss, val_accuracy

def train(train_iter, val_iter, model, classifier, params):
    """
    Function to train networks by iterating over the training set.
    :param train_iter: torchtext iterable containing batches from training set
    :param val_iter: torchtext iterable containing batches from validation set, to evaluate after each epoch
    :param model: the encoder instance
    :param classifier: a classifier instance
    :param params: object which is passed to optimizer; determines which parameters should be learned.
    """

    # Initializing optimizer and learning rate scheduler
    # Learning rate is divided by 5 for decrease in validation accuracy per epoch
    optimizer = optim.SGD(params, lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=1,
                                                     verbose=True)
    # Define a criterion to compute the loss. Note that this function has a built-in soft-max.
    criterion = nn.CrossEntropyLoss()

    val_losses = []
    val_accuracies = []
    model = model.to(device)
    classifier = classifier.to(device)

    # Loop over the specified number of epochs
    for epoch in range(1, config.epochs + 1):
        print("\nStarting training Epoch {}...\n".format(epoch))
        epoch_loss = []

        # Loop over the training set
        for i, batch in enumerate(train_iter):
            # Reset gradient attributes
            optimizer.zero_grad()
            premises, hypotheses = batch.premise, batch.hypothesis

            # Compute sentence encodings u and v
            u = model(sentences=premises[0].to(device), offsets=premises[1].to(device))
            v = model(sentences=hypotheses[0].to(device), offsets=hypotheses[1].to(device))

            #combine encodings u and v into single vector
            differences = torch.abs(u - v)
            products = torch.mul(u,v)
            final_vectors = torch.cat((u, v, differences, products), 1)

            # Do forward pass of MLP, subtract 1 from labels to get indices
            predictions = classifier(final_vectors)
            labels = batch.label -1

            # Compute loss, perform backprop & update weights.
            loss = criterion(predictions, labels.to(device))
            loss.backward()
            optimizer.step()

            epoch_loss.append( loss.item() )

            # Some boilerplate code for debugging / training on subset of epoch only, printing shapes, etcetera
            if i == 1:
                print('current model: {} \ncurrent shape of u and v: {}'.format(model, u.shape))

            # Print epoch, number of batches seen by the model, and running training loss for every _ batches.
            if i % config.print_every == 0 and i != 0:

                if config.subset == True:
                    print('Starting next Epoch..')
                    break;

                # Print loss from the last (print_every) number of batches from in epoch_loss, then reset the variable
                print('Epoch: {}, Batches: {}, Training loss: {}'.format(epoch,i,np.asarray(epoch_loss).mean()),flush=True)
                epoch_loss = []

                if config.print_val:
                    val_loss, val_accuracy = evaluate(model=model, classifier=classifier, val_iter=val_iter,
                                                  criterion=criterion)
                    print(
                        '\n------------- Validation loss after {} batches: {}. Validation accuracy: {} ------------- '.format(
                            i, val_loss, val_accuracy))

        # Evaluate model on validation set after every epoch. Check whether lr should be lowered.
        val_loss, val_accuracy = evaluate(model=model, classifier=classifier, val_iter=val_iter, criterion=criterion)
        scheduler.step(val_accuracy)

        # Store losses and accuracies for each epoch
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Save model checkpoint after each epoch. For average embeddings, we only want to save the MLP classifier weights
        if config.model == 'AVERAGE':
            PATH = "{}/{}/classifier_checkpoint_epoch_{}.pt".format(config.path, config.model, epoch)
            torch.save(classifier.state_dict(), PATH)
        # For the LSTM models we want to save checkpoints for both the classifier and the LSTM weights.
        elif config.model == 'LSTM':
            # Defining the path to save the model ...
            model_PATH = "{}/{}_bidirectional:_{}_maxpool:_{}/model_checkpoint_epoch_{}.pt".format(
                config.path, config.model, config.bidirectional, config.maxpool, epoch)
            torch.save(model.state_dict(), model_PATH)
            # Defining the path to save the corresponding classifier ...
            classifier_PATH = "{}/{}_bidirectional:_{}_maxpool:_{}/classifier_checkpoint_epoch_{}.pt".format(
                config.path, config.model, config.bidirectional, config.maxpool, epoch)
            torch.save(classifier.state_dict(), classifier_PATH)

        print('\n------------- Validation loss after epoch {}: {}. Validation accuracy: {} ------------- '.format(epoch, val_loss, val_accuracy),flush=True)

def main():
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))
    print(config.path, flush=True)

    # Build iterables, load GloVe embeddings,
    # print ('Loading data, building iterators..')
    TEXT, train_iter, val_iter, test_iter = build_iters()
    GloveVecs = TEXT.vocab.vectors
    vocab_size = GloveVecs.size(0)

    print('Initializing models...', flush=True)
    # Instantiate desired encoder model and classifier with appropriate dimensionality
    if config.model == 'AVERAGE':

        # Instantiate sentence encoder, populate with Glove embeddings.
        model = AverageEmbeddings(vocab_size=vocab_size, emb_dim=config.emb_dim)
        model.embedding.weight.data.copy_(GloveVecs)

        # Instantiate classifier
        input_dim = int(config.emb_dim * 4)
        classifier = MLPClassifier(input_dim=input_dim, hidden_dim=512)

        # Declare which weights should be optimized
        params = classifier.parameters()

    elif config.model == 'LSTM':

        # Instantiate sentence encoder, populate with Glove embeddings.
        model = LSTM_Encoder(vocab_size=vocab_size, hidden_dim=config.hidden_dim, emb_dim=config.emb_dim, num_layers=1, bidirectional=config.bidirectional, maxpool=config.maxpool)
        model.embedding.weight.data.copy_(GloveVecs)

        # Instantiate classifier
        if not config.bidirectional:
            input_dim = int(4 * config.hidden_dim)
        else:
            input_dim = int(4 * config.hidden_dim * 2)
        classifier = MLPClassifier(input_dim=input_dim, hidden_dim=512)

        # Declare which weights should be optimized
        params = list(classifier.parameters()) + list(model.encoder.parameters())
    print("Finished initializing models. Starting {}ing".format(config.mode), flush=True)

    # if mode = train, start training; otherwise, load model checkpoints and evaluate on test set
    if config.mode == "train":
        print("Beginning training.. ", flush=True)
        train(train_iter, val_iter, model, classifier, params)
        print("Finished training.. ")

    elif config.mode == "test":
        criterion = nn.CrossEntropyLoss()
        if config.model == "AVERAGE":
            classifier_path = "{}/{}/classifier_checkpoint_epoch_{}.pt".format(config.path, config.model, config.epochs)
            classifier.load_state_dict(torch.load(classifier_path))
            print("Beginning testing..", flush=True)
            test_loss, test_accuracy = evaluate(model=model, classifier=classifier, val_iter=test_iter, criterion=criterion)
        elif config.model == "LSTM":
            print("Loading classifier and model weights..", flush=True)
            classifier_path = "{}/{}_bidirectional:_{}_maxpool:_{}/classifier_checkpoint_epoch_{}.pt".format(
                config.path, config.model, config.bidirectional, config.maxpool, config.epochs)
            classifier.load_state_dict(torch.load(classifier_path))

            model_path = "{}/{}_bidirectional:_{}_maxpool:_{}/model_checkpoint_epoch_{}.pt".format(
                config.path, config.model, config.bidirectional, config.maxpool, config.epochs)
            model.load_state_dict(torch.load(model_path))

            print("Beginning testing..", flush=True)
            test_loss, test_accuracy = evaluate(model=model, classifier=classifier, val_iter=test_iter, criterion=criterion)

        print('\n------------- Loss on test set: {}. Test  accuracy: {} ------------- '.format(test_loss, test_accuracy),flush=True)

#TODO average embeddings
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden_dim', type = int, default = 2048,
                        help='Number of units in LSTM hidden layer')
    parser.add_argument('--emb_dim', type = int, default = 300,
                        help='Embedding dimension for word embeddings')
    parser.add_argument('--model', type = str, default = 'LSTM',
                        help='String to indicate preferred encoder')
    parser.add_argument('--bidirectional', type = bool, default=False,
                        help='Flag for bi-LSTM')
    parser.add_argument('--maxpool', type=bool, default=False,
                        help='Flag for maxpooling')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability for LSTM')
    parser.add_argument('--epochs', type=int, default=15,
                        help='max amount of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of datapoints per minibatch')
    parser.add_argument('--learning_rate', type = float, default = 0.1,
                        help='Learning rate for SGD')
    parser.add_argument('--print_every', type=int, default=1000,
                        help='Number of batches between loss prints')
    parser.add_argument('--subset', type=bool, default=False,
                        help='flag to only train on subset per epoch')
    parser.add_argument('--print_val', type=bool, default=False,
                        help='flag for printing eval output')
    parser.add_argument('--path', type=str, default=None,
                        help='path for saving/loading model and classifier checkpoints')
    parser.add_argument('--mode', type=str, default='train',
                        help='"train" to train model, "test" to test on checkpoints')


    config = parser.parse_args()

    main()
