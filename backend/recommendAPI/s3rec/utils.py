import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose:
            print(f"Better performance. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, rating_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, (item_list, rating_list) in enumerate(zip(user_seq, rating_seq)):
        for item, rating in zip(item_list[:-2], rating_list[:-2]):  #
            row.append(user_id)
            col.append(item)
            data.append(rating)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, rating_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, (item_list, rating_list) in enumerate(zip(user_seq, rating_seq)):
        for item, rating in zip(item_list[:-1], rating_list[:-1]):  #
            row.append(user_id)
            col.append(item)
            data.append(rating)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_submission(user_seq, rating_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, (item_list, rating_list) in enumerate(zip(user_seq, rating_seq)):
        for item, rating in zip(item_list[:], rating_list[:]):  #
            row.append(user_id)
            col.append(item)
            data.append(rating)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_submission_file(data_file, preds):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        "output/submission.csv", index=False
    )

def __save_labels(output_dir, encoder, name):
    le_path = os.path.join(output_dir, name + "_classes.npy")
    np.save(le_path, encoder.classes_)

# def __preprocessing()


def get_user_seqs(args, is_train = True):
    # TODO get the data from DataBase # rating_df = db.query ...
    rating_df = pd.read_csv(args.data_file)
    
    # labele encoding
    le = LabelEncoder()
    if is_train:
        raw_item_list = rating_df["item"].unique().tolist() + [-99999] # "unknown" -> -99999
        le.fit(raw_item_list)
        __save_labels(args.output_dir, le, "item")
    else:
        label_path = os.path.join(args.output_dir, "item" + "_classes.npy")
        le.classes_ = np.load(label_path)
    
    rating_df["item"] = le.transform(rating_df["item"])

    # lines = rating_df.groupby("user")["item"].apply(list)
    lines_item = rating_df.groupby("user")["item"].apply(list)
    lines_rating = rating_df.groupby("user")["rating"].apply(list)

    # user_seq = []
    # item_set = set()
    user_seq = []
    rating_seq = []
    item_set = set()

    # for line in lines:
        # items = line
        # user_seq.append(items)
        # item_set = item_set | set(items)
    for line_item, line_rating in zip(lines_item, lines_rating):
        items = line_item
        user_seq.append(items)
        ratings = line_rating
        rating_seq.append(ratings)
        item_set = item_set | set(items)
    
    # max_item = max(item_set)
    # num_users = len(lines)
    # num_items = max_item + 2
    max_item = max(item_set)
    num_users = len(lines_item)
    num_items = max_item + 2
    
    # valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    # test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    valid_rating_matrix = generate_rating_matrix_valid(user_seq, rating_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, rating_seq, num_users, num_items)
    submission_rating_matrix = generate_rating_matrix_submission( #### 이 함수도 안쓸것 같습니다.
        user_seq, rating_seq, num_users, num_items 
    )
    return (
        user_seq,
        rating_seq, ############ NEW ############
        max_item,
        valid_rating_matrix,
        test_rating_matrix,
        submission_rating_matrix,
    )


def get_user_seqs_long(args, is_train = True):
    rating_df = pd.read_csv(args.data_file)

    # labele encoding
    le = LabelEncoder()
    if is_train:
        raw_item_list = rating_df["item"].unique().tolist() + [-99999] # "unknown" -> -99999
        le.fit(raw_item_list)
        __save_labels(args.output_dir, le, "item")
    else:
        label_path = os.path.join(args.output_dir, "item" + "_classes.npy")
        le.classes_ = np.load(label_path)

    # lines = rating_df.groupby("user")["item"].apply(list)
    # user_seq = []
    # long_sequence = []
    # item_set = set()
    # for line in lines:
    #     items = line
    #     long_sequence.extend(items)
    #     user_seq.append(items)
    #     item_set = item_set | set(items)
    # max_item = max(item_set)
    # return user_seq, max_item, long_sequence

    lines_item = rating_df.groupby("user")["item"].apply(list)
    lines_rating = rating_df.groupby("user")["rating"].apply(list)
    user_seq = []
    rating_seq = []
    long_sequence = [] 
    item_set = set()
    for line_item, line_rating in zip(lines_item, lines_rating):
        items = line_item
        ratings = line_rating
        long_sequence.extend(items)
        user_seq.append(items)
        rating_seq.append(ratings)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, rating_seq, max_item, long_sequence
    


def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
