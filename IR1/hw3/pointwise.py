import dataset
import pickle as pkl
import ranking as rnk
import evaluate as evl
import numpy as np
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import scipy.stats as st




os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class PointWiseLTR(nn.Module):
    """
    This class implements a pointwise LTR model.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initialize PLTR object.

        Args:
          batch_size:
        """
        super(PointWiseLTR, self).__init__()

        # Initialize layers
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_size/2), 1)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        """
        out = self.net(x)
        return out


class bigDataset(Dataset):
    """
    Dataset class for the assignment data. Returns row of feature matrix
    and corresponding label.
    """
    def __init__(self, data):
        self.x = data.feature_matrix
        self.y = data.label_vector

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        data_val = self.x[index,:]
        target = self.y[index].reshape((-1))
        return data_val, target


class EarlyStopping:
    """
    Stops training early if validation loss does not decrease anymore, within
    a predefined number of steps (patience).
    Partially adapted from EarlyStopping class in pytorchtools package.
    (https://github.com/Bjarten/early-stopping-pytorch)
    """
    def __init__(self, patience=5, delta=0.00001, save_path="checkpoint.pt"):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_loss = 0
        self.delta = delta
        self.save_path = save_path

    def __call__(self, ndcg, model):
        # If there is no best_score yet assign initial loss
        if self.best_score is None:
            self.best_score = ndcg
            torch.save(model.state_dict(), self.save_path)
        # If loss increased start the patience counter
        elif ndcg < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        # If the loss decreased set counter to 0 and save model
        else:
            self.best_score = ndcg
            torch.save(model.state_dict(), self.save_path)
            self.counter = 0


def evaluate_model(model, data, criterion, val_loader):
    """
    Evaluate the model on the validation set. Returns all evaluation
    results and the average validation loss.
    """
    val_loss, scores = [], []
    # Loop through validation data
    for step, (batch_x, batch_y) in enumerate(val_loader):
        batch_x, batch_y = batch_x.float(), batch_y.float()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        val_loss.append(loss.item())
        # Save scores per batch
        add_output = output.view(-1).detach().numpy()
        scores.append(add_output)

    # Get results to return
    scores = np.concatenate(scores)
    results = evl.evaluate(data, scores)
    val_loss = np.asarray(val_loss).mean()
    return results, val_loss, scores

def test_model(model, criterion, data):
    """
    Evaluate the model on the test set.
    """
    test_set = bigDataset(data)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    test_loss, scores = [], []
    # Loop through test data
    for step, (batch_x, batch_y) in enumerate(test_loader):
        batch_x, batch_y = batch_x.float(), batch_y.float()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        test_loss.append(loss.item())
        # Save scores per batch
        add_output = output.view(-1).detach().numpy()
        scores.append(add_output)

    # Get results to return
    test_scores = np.concatenate(scores)
    test_results = evl.evaluate(data, test_scores)
    test_loss = np.asarray(test_loss).mean()
    return test_results, test_loss, test_scores


def plot_performance(val_loss, ndcg):
    """
    Plot the performance measures of the model.
    """
    epochs = list(range(1, len(val_loss) + 1))
    epochs = [x*config.batch_multiple for x in epochs]
    x = epochs
    y = [val_loss, ndcg]
    labels = ['Validation Loss', 'NDCG']
    for y_arr, label in zip(y, labels):
        plt.plot(x, y_arr, label=label)
    plt.title("Pointwise LTR, NDCG and validation loss")
    plt.xlabel('Batches of size {}'.format(config.batch_size))
    plt.xlim(left=3500)
    plt.ylim(top=1.20)
    plt.ylabel('Loss, NDCG')
    plt.legend()
    plt.show()


def plot_distributions(labels, scores, name):
    """
    Plot score ditributions.
    """
    plt.clf()
    plt.style.use('seaborn')
    plt.hist(labels, density=True, bins=5, label='actual grades', alpha=1)
    plt.hist(scores, density=True, bins=60, label='scores', alpha=0.5)
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    plt.ylabel('probability')
    plt.xlabel('relevance')
    plt.legend(loc='upper right')
    plt.savefig('./plots/{}'.format(name))


def train(data):
    #Create dataloader objects for train and validation data
    train_set = bigDataset(data.train)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_set = bigDataset(data.validation)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    # Initialize model, loss, optimizer
    model = PointWiseLTR(data.num_features, config.hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    criterion = nn.MSELoss()
    val_losses, val_ndcg, val_scores = [], [], []

    # Initialize early stopper
    stopper = EarlyStopping(config.patience, config.delta, "checkpoint.pt")

    for epoch in range(1, config.epochs + 1):
        # Go over entire train dataset per epoch
        print("------ Training epoch ", epoch)
        epoch_loss = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.float(), batch_y.float()
            # Reset gradients
            optimizer.zero_grad()
            # Do forward step, calculate loss, do backward step and optimize
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            # Check whether we want to produce a plot later
            # if so, evaluate model on validation set for every x batches and record NDCG and loss;
            if step % config.batch_multiple == 0:
                val_results, val_loss, val_score = evaluate_model(model, data.validation, criterion, val_loader)

                print("NDCG:", val_results['ndcg'])
                print("Validation loss:", np.asarray(val_loss).mean())

                val_scores.append(val_score)
                val_losses.append(np.asarray(val_loss).mean())
                val_ndcg.append(val_results['ndcg'][0])

                stopper(val_results['ndcg'][0], model)
                if stopper.early_stop:
                    print("Early stopping")
                    break

        print("Train loss:", np.asarray(epoch_loss).mean())

        stopper(val_results['ndcg'][0], model)
        if stopper.early_stop:
            print("Early stopping")
            break


    #e Evaluate model on test set
    if config.test_model:
        print('Testing model on test set...')
        test_results, test_loss, test_scores = test_model(model, criterion, data.test)
        print("NDCG on test set:", test_results['ndcg'][0])
        print("Test loss:", np.asarray(test_loss).mean())

    # Compare distributions of the scores with the distributions of the actual grades
    val_results, val_loss, val_scores = evaluate_model(model, data.validation, criterion, val_loader)
    
    if config.produce_plot:
        val_labels = data.validation.label_vector
        test_labels = data.test.label_vector
        train_labels = data.train.label_vector
        plot_distributions(val_labels, val_scores, "val_distribution")
        plot_distributions(test_labels, test_scores, "test_distribution")
    return val_losses, val_ndcg


    return val_losses, val_ndcg


def main():
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    val_losses, val_ndcg = train(data)

    if config.produce_plot:
        plot_performance(val_losses, val_ndcg)



if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_units', type = int, default = 100,
                      help='Number of units in hidden layer')
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                      help='Learning rate')
    parser.add_argument('--batch_size', type = int, default = 128,
                      help='Batch size to run trainer.')
    parser.add_argument('--epochs', type=int, default=40,
                        help='max amount of epochs to train for')
    parser.add_argument('--delta', type=float, default=0.0001,
                        help='difference between epochs for early stopping')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience parameter for early stopping')
    parser.add_argument('--batch_multiple', type=int, default=50,
                         help='number of batches between evaluation measurements')
    parser.add_argument('--produce_plot', type=bool, default=False,
                        help='flag to suppress plotting of validation')
    parser.add_argument('--test_model', type=bool, default=False,
                        help='flag to prevent model testing')
    config = parser.parse_args()

    main()

# Testen voor learning rate: 0.1 - 0.000001
# Testen voor batch size: 32 - 64 - 128 - 256
# Testen voor hidden nodes: 50 -100 - 150
