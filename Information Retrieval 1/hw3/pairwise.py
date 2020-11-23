import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import sys
import argparse
import itertools
import numpy as np
import math
from tqdm import tqdm
import evaluate as evl
import torch
import torch.nn as nn
import pickle as pkl

import matplotlib.pyplot as plt
import os

from pointwise import EarlyStopping

class PairWiseLTR(nn.Module):
    """
    This class implements a pairtwise LTR model.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initialize PLTR object.
        Args:
          batch_size:
        """
        super(PairWiseLTR, self).__init__()

        # Initialize layers
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        """
        out = self.net(x)
        return out


def get_Sij(diff):
    """
    Get correct Sij value for true scores.
    """
    if diff > 0:
        return 1
    elif diff < 0:
        return -1
    else:
        return 0


def get_pairs(scores_with_ids, final_pairs, big_S):
    """
    Compare pairs of documents and save outcomes per doc.
    """
    prev_id = 0
    id_list = []
    # Get all possible pred pairs i,j.
    for pair in itertools.permutations(scores_with_ids, r=2):
        cur_id = pair[0][0]
        # Check if we're still at same doc i
        if cur_id == prev_id:
            diff = pair[0][1] - pair[1][1]
            if big_S:
                diff = get_Sij(diff)
            id_list.append(diff)
        # Add all lambda_ij's to lambda_i list
        else:
            final_pairs.append(id_list)
            id_list = [pair[0][1] - pair[1][1]]
        prev_id = cur_id
    final_pairs.append(id_list)
    return final_pairs


def compute_loss(pred_scores, true_scores):
    if (pred_scores.size(0) > 1):
        sigma = 1 # hyper parameter
        pred_scores = pred_scores.view(-1)        
        true_scores = true_scores.view(-1)
        z, s = [], []

        # Keep doc ids for lambda calculation
        if config.sped_up:
            pred_with_ids = [(i, pred.item()) for i, pred in enumerate(pred_scores)]
            true_with_ids = [(i, true.item()) for i, true in enumerate(true_scores)] 
            
            # Get all values for s_i - s_j
            z = get_pairs(pred_with_ids, z, False)
            # Get all values for Sij
            s = get_pairs(true_with_ids, s, True)
            # Convert to tensors
            z = torch.FloatTensor(z)
            s = torch.FloatTensor(s)

        else:
            # Get all possible pairs i,j.
            for pair in itertools.combinations(pred_scores, r=2):
                z.append(pair[0]-pair[1]) #predicted difference between si en sj

            for pair in itertools.combinations(true_scores, r=2):
                s.append(pair[0]-pair[1]) #true difference between si en sj
            z = torch.stack(z)
            s = torch.stack(s)
            s[s > 0] = 1
            s[s < 0] = -1

        # Calculate lambda without grads for sped up version
        if config.sped_up is True:
            with torch.no_grad():
                lambda_ij = sigma*(0.5*(1-s) - (1/(1+torch.exp(sigma*z))))
                lambda_i = torch.sum(lambda_ij, dim=1, keepdim=True)
                loss = lambda_i
        # Else just calculate the loss
        else:
            loss = 0.5*(1-s)*sigma*z + torch.log(1+torch.exp(-sigma*z))
            loss = torch.sum(loss)

    # If a query has only one relevant doc, loss is 0
    else:
        loss = torch.zeros(1, requires_grad=True)
        loss = loss.view(1,1)
    return(loss)


def evaluate_model(model, data):
    x = data.feature_matrix
    y = data.label_vector
    total_loss, total_scores = [], []

    # Go over entire validation dataset
    for q in range(0, data.num_queries()):
        # Get batch for training
        s_i, e_i = data.query_range(q)
        query_x = x[s_i:e_i]
        query_x = torch.Tensor(query_x)
        query_y = y[s_i:e_i]
        query_y = torch.Tensor(query_y).view(-1,1)

        # Get scores and calculate loss
        scores = model(query_x)
        loss = compute_loss(scores, query_y)
        if config.sped_up:
            batch_size = query_y.shape[0]
            loss = torch.mean(loss)

        # Save scores and loss
        total_loss.append(loss.item())
        scores = scores.view(-1).detach().numpy()
        total_scores.append(scores)

    loss = np.asarray(total_loss).mean()
    total_scores = np.concatenate(total_scores)

    results = evl.evaluate(data, total_scores)
    return results, loss



def plot_performance(val_loss, ndcg, arr, convergence):
    """
    If convergence is True, only plot the loss (AQ3.1) to assess convergence 
    ; otherwise, plot ARR and NDCG (AQ3.2)
    """
    if convergence==True:
        num_measurements = list(range(1, len(val_loss) + 1))
        x = num_measurements
        y = [val_loss]
        labels = ['validation loss']
        for y_arr, label in zip(y, labels):
            plt.plot(x, y_arr, label=label)
        plt.title = "Pairwise LTR loss"
        plt.legend()
        plt.show()
    else:
        num_measurements = list(range(1, len(val_loss) + 1))
        x = num_measurements
        y = [ndcg, arr]
        labels = ['NDCG', 'ARR']
        for y_arr, label in zip(y, labels):
            plt.plot(x, y_arr, label=label)
        plt.title="Pairwise LTR: NDCG, ARR"
        plt.legend()
        plt.show()


def train(data):
    # Get data info
    input_size = data.num_features
    data_size = data.train.num_docs()

    # Get data
    train_x = data.train.feature_matrix
    train_y = data.train.label_vector

    # Initialize model, loss, optimizer
    model = PairWiseLTR(input_size, config.hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

    # Initialize early stopper
    stopper = EarlyStopping(config.patience, config.delta, "checkpoint.pt")

    #lists to store losses and metrics for plotting
    val_losses, val_ndcg, val_arr, train_losses = [], [], [], []


    for epoch in range(config.epochs):
        print("epoch ", epoch)
        # Go over train dataset
        for q in tqdm(range(0, data.train.num_queries())):
            # Get batch for training
            s_i, e_i = data.train.query_range(q)
            query_x = train_x[s_i:e_i]
            query_x = torch.Tensor(query_x)
            query_y = train_y[s_i:e_i]
            query_y = torch.Tensor(query_y).view(-1,1)

            # Do forward step, calculate loss, do backward step and optimize
            optimizer.zero_grad()
            scores = model(query_x)
            loss = compute_loss(scores, query_y)
            # Backprop loss
            if config.sped_up:
                batch_size = query_y.shape[0]
                # print("LOSS", scores * loss)
                scores.backward(loss)
            else:
                # print("LOSS", loss)
                loss.backward()
            optimizer.step()

            #flag to suppress printed updates for loss, NDCG and ERR needed for plot
            if config.produce_plot:
                if q % config.batch_multiple == 0:
                    train_losses.append(torch.mean(loss).item())

                    val_results, val_loss = evaluate_model(model, data.validation)
                    print("validation loss:", val_loss.item())
                    print("validation NDCG:", val_results['ndcg'])
                    print("validation ARR:", val_results['relevant rank'])
                    val_losses.append(np.asarray(val_loss).mean())
                    val_ndcg.append(val_results['ndcg'][0])
                    val_arr.append(val_results['relevant rank'][0])

                    stopper(val_results['ndcg'][0], model)
                    if stopper.early_stop:
                        print("Early stopping")
                        break

        # print("train loss:", loss.item())
        # Evaluate model on validation set
        val_results, val_loss = evaluate_model(model, data.validation)
        print("validation loss:", val_loss.item())
        print("validation ndcg:", val_results['ndcg'])

        # Stop training if early stopping criteria have been met
        stopper(val_results['ndcg'][0], model)
        if stopper.early_stop:
            break

    #evaluate model on test set
    if config.test_model:
        print('Testing model on test set...')
        test_results, test_loss = evaluate(model, data.test)
        print("NDCG on test set:", test_results['ndcg'][0])
        print("ERR on test set:", test_results['ERR'][0])
        print("Test loss:", np.asarray(test_loss).mean())

    return val_losses, val_ndcg, val_arr, train_losses




def main():
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    val_losses, val_ndcg, val_arr, train_losses = train(data)

    #append val_losses and train_losses to list, save to disk
    losses = []
    losses.append(val_losses)
    losses.append(train_losses)
    with open('./pairwise_losses/pairwise_losses_spedup_{}'.format(config.sped_up), 'wb') as f:
        pkl.dump(losses, f)

    if config.produce_plot:
        plot_performance(val_losses, val_ndcg, val_arr, config.convergence)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_units', type = int, default = 100,
                      help='Number of units in hidden layer')
    parser.add_argument('--learning_rate', type = float, default = 0.0001,
                      help='Learning rate')
    parser.add_argument('--batch_size', type = int, default = 128,
                      help='Batch size to run trainer.')
    parser.add_argument('--epochs', type=int, default=40,
                        help='max amount of epochs to train for')
    parser.add_argument('--delta', type=float, default=0.0001,
                        help='difference between epochs for early stopping')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience parameter for early stopping')
    parser.add_argument('--sped_up', type=bool, default=False,
                        help='use sped up Ranknet')
    parser.add_argument('--batch_multiple', type=int, default=50,
                         help='number of batches between evaluation measurements')
    parser.add_argument('--produce_plot', type=bool, default=False,
                        help='flag to suppress plotting of metrics and loss on validation')
    parser.add_argument('--convergence', type=bool, default=False,
                        help='flag to switch between plotting of loss or ARR & NDCG')
    parser.add_argument('--test_model', type=bool, default=False,
                        help='flag to evaluate model on test set')

    config = parser.parse_args()

    main()