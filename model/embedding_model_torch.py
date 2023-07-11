import numpy as np
from load_data import load_data_as_numpy, split_test_and_train
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from preprocess_for_embeddings import preprocess
from sklearn.decomposition import PCA
from scratch_file import id_to_teamname_and_record

import torch

class FFModelWithEmbeddings(torch.nn.Module):
    def __init__(self, batch_size, team_id_max, in_features=206):
        super().__init__()
        self.logdir="logs/FFModelWithEmbeddings/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.batch_size = batch_size
        self.E = torch.nn.Embedding(team_id_max, 8)
        self.reshape_layer = torch.nn.Flatten()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.001)

        
    # Inputs is the vector of stats (no date, no team) normalized
    # TeamIds corresponds to a unique season/team combo
    def call(self, inputs, team_ids):
        # get embeddings
        team_and_year_embeddings = self.E(team_ids)
        # flatten embeddings
        team_and_year_embeddings = self.reshape_layer(team_and_year_embeddings)
        # concatenate embeddings and inputs
        inputs = torch.cat((inputs, team_and_year_embeddings), 1)
        # run through model
        return self.model(inputs)
        
    def mae_loss(self, y_pred, y_true):
        return torch.nn.functional.l1_loss(y_true, y_pred)
    
    def mse_loss(self, y_pred, y_true):
        return torch.nn.functional.mse_loss(y_pred, y_true)

def train(model, train_features, train_labels, train_ids):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: avg loss
    """
    total_loss = 0
    batch_size = 32
    idx = 0
    while (idx + 1) * batch_size < len(train_labels):
        model.optimizer.zero_grad()
        preds = model.call(train_features[idx * batch_size:min(idx * batch_size + batch_size, len(train_ids))],
                           train_ids[idx * batch_size:min(idx * batch_size + batch_size, len(train_ids))])[:,0]
        loss = model.mae_loss(preds, train_labels[idx * batch_size:min(idx * batch_size + batch_size, len(train_ids))])
        total_loss += loss
        idx += 1
        loss.backward()
        model.optimizer.step()
    avg_loss = total_loss / idx
    return avg_loss

def test(model, test_features, test_labels, test_ids, batch_size=32, return_preds_and_labels=False):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: avg loss
    """
    total_loss = 0
    total_l1_error = 0
    total_l2_error = 0
    #batch inputs and labels
    idx = 0
    seen_examples = 0
    all_preds = []
    all_labels = []

    while (idx + 1) * batch_size < len(test_labels):
        preds = model.call(test_features[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))],
                               test_ids[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))])[:,0]
        all_preds.extend(list(preds.detach().numpy()))
        all_labels.extend(list(test_labels[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))].detach().numpy()))
        loss = model.mae_loss(preds, test_labels[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))])
        seen_examples += len(preds)
        total_l1_error += torch.nn.functional.l1_loss(preds, test_labels[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))])
        total_l2_error += torch.nn.functional.mse_loss(preds, test_labels[idx * batch_size:min(idx * batch_size + batch_size, len(test_ids))])
        total_loss += loss
        idx += 1

    avg_loss = total_loss / (idx)
    avg_err = total_l1_error / (idx)
    avg_l2_err = total_l2_error / (idx)
    
    if return_preds_and_labels:
        return avg_loss, avg_err, all_preds, all_labels
    return avg_loss, avg_err, avg_l2_err

def plot_embedding(team_ids, labels, model):
    vecs = model.E(np.array(team_ids, dtype=int))
    pca_two_dim = PCA(n_components=2)
    pca_one_dim = PCA(n_components=1)
    basis_vecs_two_dim = pca_two_dim.fit_transform(vecs)
    basis_vecs_one_dim = pca_one_dim.fit_transform(vecs)
    two_d_projections = []
    one_d_projections = []
    for idx in range(len(basis_vecs_two_dim)):
        two_d_projections.append(basis_vecs_two_dim[idx])
        one_d_projections.append(basis_vecs_one_dim[idx])
    x = [x for [x, y] in two_d_projections]
    y = [y for [x, y] in two_d_projections]
    cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=0, vmax=1)  
    plt.scatter([id_to_teamname_and_record[label][1] for label in range(1610612737, 1610612767)], one_d_projections)
    plt.show()
    plt.scatter(x, y, c=[id_to_teamname_and_record[label][1] for label in range(1610612737, 1610612767)], cmap=cmap, norm=norm)
    plt.show()
    plt.scatter([id_to_teamname_and_record[label][1] for label in range(1610612737, 1610612767)], [x_i**2 + y_i**2 for (x_i, y_i) in zip(x, y)])
    for idx, point in enumerate(zip(x, y)):
        plt.annotate("{}, {}".format(id_to_teamname_and_record[labels[idx]][0], id_to_teamname_and_record[labels[idx]][1]), point)
    plt.show()

if __name__ == "__main__":
    train_features, train_labels, test_features, test_labels, train_ids, test_ids, team_map = preprocess()
    # get mean and std of train features
    mean = torch.mean(train_features, dim=0)
    std = torch.std(train_features, dim=0)
    # normalize train and test features
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std
    # convert above to torch tensor of ints
    train_ids, test_ids = torch.tensor(train_ids, dtype=torch.int64), torch.tensor(test_ids, dtype=torch.int64)
    model = FFModelWithEmbeddings(32, max(torch.max(train_ids), torch.max(test_ids)) + 1) #make sure there are enough embeddings
    for epoch in range(20):
        print(epoch)
        print(train(model, train_features, train_labels, train_ids))
        print(test(model, test_features, test_labels, test_ids))
    avg_loss, avg_error, avg_l2_error = test(model, test_features, test_labels, test_ids)
    embedding_ids = []
    for team_id in range(1610612737, 1610612767):
        embedding_ids.append(team_map[(2018, team_id)])
        # for year in range(2007, 2021):
        #     if (year, team_id) in team_map:
        #         embedding_ids.append(team_map[(year, team_id)])
    plot_embedding(embedding_ids, range(1610612737, 1610612767), model)
    test_features, test_labels, ids, team_map, spreads = preprocess(for_testing=True)
    ids = np.ndarray.astype(ids, int)
    loss, err, preds, labels = test(model, test_features, test_labels, ids, return_preds_and_labels=True)