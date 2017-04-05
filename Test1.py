"""
Copyright 2016 Ronald J. Nowling
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from collections import defaultdict
import random
from DataHelper import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

import numpy as np
import scipy.sparse as sp



def create_training_sets(ratings, n_training, n_testing):
    print("Creating user movie-interaction lists")

    user_interactions = defaultdict(set)
    max_movie_id = 0
    for r in ratings:
        user_interactions[r.user_id].add(r.movie_id)
        max_movie_id = max(max_movie_id, r.movie_id)

    user_interactions = list(user_interactions.values())
    sampled_indices = random.sample(range(len(user_interactions)), n_training + n_testing)

    users = []
    movies = []
    interactions = []
    for new_user_id, idx in enumerate(sampled_indices[:n_training]):
        users.extend([new_user_id] * len(user_interactions[idx]))
        movies.extend(user_interactions[idx])
        interactions.extend([1.] * len(user_interactions[idx]))

    n_movies = max_movie_id + 1
    training_matrix = sp.coo_matrix((interactions, (users, movies)),
                                    shape=(n_training, n_movies)).tocsr()

    users = []
    movies = []
    interactions = []
    for new_user_id, idx in enumerate(sampled_indices[n_training:]):
        users.extend([new_user_id] * len(user_interactions[idx]))
        movies.extend(user_interactions[idx])
        interactions.extend([1.] * len(user_interactions[idx]))

    n_movies = max_movie_id + 1
    testing_matrix = sp.coo_matrix((interactions, (users, movies)),
                                   shape=(n_testing, n_movies)).tocsr()

    print(training_matrix.shape, testing_matrix.shape)

    return training_matrix, testing_matrix


def train_and_score(metric, training, testing, ks):
    print("Training and scoring")
    scores = []
    knn = NearestNeighbors(metric=metric, algorithm="brute")
    knn.fit(training)
    for k in ks:
        print("Evaluating for", k, "neighbors")
        neighbor_indices = knn.kneighbors(testing,
                                          n_neighbors=k,
                                          return_distance=False)

        all_predicted_scores = []
        all_labels = []
        for user_id in range(testing.shape[0]):
            user_row = testing[user_id, :]

            interaction_indices = user_row.nonzero()
            interacted = set(interaction_indices)
            non_interacted = set(range(testing.shape[1])) - interacted

            n_samples = min(len(non_interacted), len(interacted))
            sampled_interacted = random.sample(interacted, n_samples)
            sampled_non_interacted = random.sample(non_interacted, n_samples)

            indices = list(sampled_interacted)
            indices.extend(sampled_non_interacted)
            labels = [1] * n_samples
            labels.extend([0] * n_samples)

            neighbors = training[neighbor_indices[user_id, :], :]
            predicted_scores = neighbors.mean(axis=0)
            for idx in indices:
                all_predicted_scores.append(predicted_scores[0, idx])
            all_labels.extend(labels)

        print(len(all_labels), len(all_predicted_scores))

        auc = roc_auc_score(all_labels, all_predicted_scores)

        print("k", k, "AUC", auc)


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ratings-fl",
                        type=str,
                        required=True,
                        help="Ratings file")

    parser.add_argument("--training",
                        type=int,
                        default=10000,
                        help="Number of training samples")

    parser.add_argument("--testing",
                        type=int,
                        default=1000,
                        help="Number of testing samples")

    parser.add_argument("--metric",
                        type=str,
                        choices=["euclidean", "cosine"],
                        default="euclidean",
                        help="Distance metric")

    parser.add_argument("--ks",
                        type=int,
                        nargs="+",
                        required=True,
                        help="Number of neigbhors")

    return parser.parse_args()


if __name__ == "__main__":
    MyData = LoadMovieLens100k('Datas/ml-100k/u.data')
    print(MyData.head())
    n_users = MyData.user_id.unique().shape[0]
    n_items = MyData.item_id.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
    train_data, test_data = SpiltData(MyData, SpiltRate=0.25)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    # Create two user-item matrices, one for training and another for testing
    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    train_and_score('cosine',
                    train_data_matrix,
                    test_data_matrix,
                    [25])