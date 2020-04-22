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
import ranking as rnk


import torch
import torch.nn as nn

from pointwise import EarlyStopping

class ListWiseLTR(nn.Module):
    """
    This class implements a listtwise LTR model.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initialize PLTR object.

        Args:
          batch_size:
        """
        super(ListWiseLTR, self).__init__()

        # Initialize layers
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        """
        out = self.net(x)
        return out


def evaluate_model(model, data, eval_metric):
    """
    Evaluate the model on the validation set.
    """
    x = data.feature_matrix
    y = data.label_vector
    total_loss, total_scores = [], []

    # Go over entire validation dataset
    for q in tqdm(range(0, data.num_queries())):
        # Get batch for training
        s_i, e_i = data.query_range(q)
        query_x = x[s_i:e_i]
        query_x = torch.Tensor(query_x)
        query_y = y[s_i:e_i]
        query_y = torch.Tensor(query_y).view(-1,1)

        # Get scores and calculate loss
        scores = model(query_x)
        loss = compute_loss(scores, query_y, eval_metric)
        loss = torch.mean(loss)

        # Save scores and loss
        total_loss.append(loss.item())
        scores = scores.view(-1).detach().numpy()
        total_scores.append(scores)

    loss = np.asarray(total_loss).mean()
    total_scores = np.concatenate(total_scores)

    results = evl.evaluate(data, total_scores)
    return results, loss


def get_Sij(diff):
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
        curr_id = pair[0][0]
        if curr_id == prev_id:
            diff = pair[0][1] - pair[1][1]
            if big_S:
                diff = get_Sij(diff)
            id_list.append(diff)
        else:
            final_pairs.append(id_list)
            id_list = [pair[0][1] - pair[1][1]]
        prev_id = curr_id
    final_pairs.append(id_list)
    return final_pairs


def get_score(pred_scores, true_ranking):
    """
    Get the NDCG or ERR score of predicted scores vs true scores.
    """
    pred_ranking, _ = rnk.rank_and_invert(pred_scores)
    if config.eval_metric == 'ndcg':
        score = evl.ndcg_at_k(pred_ranking, true_ranking, 0)
    elif config.eval_metric == 'err':
        score = evl.ERR(pred_ranking, true_ranking)
    else:
        raise Exception('Evaluation metric should be either ndcg or err.')
    return score

def calculate_delta(swap_scores, curr_id, compare_id, true_ranking, before_irm):
    """
    Calculate delta_irm(i,j).
    """
    # Swap a doc in the ranking
    swap_scores[curr_id], swap_scores[compare_id] = swap_scores[compare_id], swap_scores[curr_id]
    # Compute new score and get the difference with true score
    after_irm = get_score(np.asarray(swap_scores), true_ranking)
    delta_ij = abs(before_irm - after_irm)
    return delta_ij


def get_IRM(pred_scores, true_scores):
    """
    Get an array with all delta_IRM(i,j) for all doc combinations 
    corresponding to a query.
    """
    final_irm = []
    # When all relevances are 0, we set all delta irm to 1
    if sum(true_scores) != 0:
        true_ranking, _ = rnk.rank_and_invert(true_scores)
        before_irm = get_score(pred_scores, true_ranking)
        delta_irm = []
        prev_id = 0
        # Loop through all permutations of the docs
        for pair in itertools.permutations(enumerate(pred_scores), r=2):
            swap_scores = list(pred_scores)
            curr_id = pair[0][0]
            compare_id = pair[1][0]
            if curr_id == prev_id:
                delta_ij = calculate_delta(swap_scores, curr_id, compare_id, true_ranking, before_irm)
                delta_irm.append(delta_ij)
            else:
                final_irm.append(delta_irm)
                delta_ij = calculate_delta(swap_scores, curr_id, compare_id, true_ranking, before_irm)
                delta_irm = [delta_ij]
            prev_id = curr_id
        final_irm.append(delta_irm)
    else:
        final_irm = np.ones(((len(true_scores)), len(true_scores)-1))
    return final_irm


def compute_loss(pred_scores, true_scores, eval_metric):
    if (pred_scores.size(0) > 1):
        sigma = 1 # hyper parameter
        pred_scores = pred_scores.view(-1)
        true_scores = true_scores.view(-1)
        z, s = [], []

        # Retrieve IRM values for all i,j combinations

        irm = get_IRM(pred_scores.detach().numpy(), true_scores.detach().numpy())
        delta_IRM = torch.FloatTensor(irm)

        # Put all scores in list with corresponding id
        pred_with_ids = [(i, pred.item()) for i, pred in enumerate(pred_scores)]
        true_with_ids = [(i, true.item()) for i, true in enumerate(true_scores)]

        # Get all values for s_i - s_j
        z = get_pairs(pred_with_ids, z, False)
        # Get all values for Sij
        s = get_pairs(true_with_ids, s, True)
        # Convert to tensors
        z = torch.FloatTensor(z)
        s = torch.FloatTensor(s)

        # Calculate lambda_i
        with torch.no_grad():
            lambda_ij = sigma*(0.5*(1-s) - (1/(1+torch.exp(sigma*z))))
            lambda_irm = lambda_ij * delta_IRM
            lambda_i = torch.sum(lambda_irm, dim=1, keepdim=True)
            loss = lambda_i

    # If a query has only one relevant doc, loss is 0
    else:
        loss = torch.zeros(1, requires_grad=True)
        loss = loss.view(1,1)
    return(loss)




def train(data):
    # Get data info
    input_size = data.num_features

    # Get data
    train_x = data.train.feature_matrix
    train_y = data.train.label_vector

    # Initialize model, loss, optimizer
    model = ListWiseLTR(input_size, config.hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

    # Initialize early stopper
    stopper = EarlyStopping(config.patience, config.delta, "checkpoint.pt")

    val_losses, val_ndcg, val_losses_test = [], [], []

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
            loss = compute_loss(scores, query_y, config.eval_metric)
            scores.backward(loss)
            optimizer.step()

            if q % config.batch_multiple == 0:
                metric = config.eval_metric
                val_results, val_loss = evaluate_model(model, data.validation, metric)
                print("Query:", q, " -- validation score:", val_results[metric])

                # Save data for plotting
                val_losses.append(np.asarray(val_loss).mean())
                val_losses_test.append(np.asarray(val_loss).sum())
                val_ndcg.append(val_results[metric][0])

                # Chec kfor convergence
                stopper(val_results[metric][0], model)
                if stopper.early_stop:
                    print("Early stopping")
                    break

        if stopper.early_stop:
            print(val_losses)
            print(val_ndcg)
            break

        # Evaluate model on validation set
        val_results, val_loss = evaluate_model(model, data.validation, config.eval_metric)
        print("validation loss:", val_loss.item())
        print("validation ndcg:", val_results['ndcg'])

    print('Testing model on test set...')
    test_results, test_loss, test_scores = test_model(model, criterion, data.test)
    print("NDCG on test set:", test_results['ndcg'][0])
    print("Test loss:", np.asarray(test_loss).mean())




def main():
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    train(data)


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
    parser.add_argument('--eval_metric', type=str, default='ndcg',
                        help='metric for ranking evaluation (ndcg or err)')
    parser.add_argument('--batch_multiple', type=int, default=500.
                        help='Amount of processed batches after which to evaluate.')

    config = parser.parse_args()

    main()
