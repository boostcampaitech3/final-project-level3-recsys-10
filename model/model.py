import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# AutoRec HP
num_hidden = 100
num_items = 80

class AutoRec(nn.Module):
    def __init__(self, num_hidden, num_items, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_items, num_hidden)
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Linear(num_hidden, num_items)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mat):
        hidden = self.dropout(self.sigmoid(self.encoder(mat)))
        pred = self.decoder(hidden)
        
        return pred

# TODO 상대경로를 어떻게 찾나.. 절대경로 에반대
def get_model(model_path: str = "model/autorec_crawling.pt")-> AutoRec:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoRec(num_hidden, num_items).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def _transform(data : dict):
    beer_mapping = pd.read_csv('data/ratebeer_label_encoding.csv')
    x_test = [0 for _ in range(beer_mapping.shape[0])]
    for key, value in data.items():
        encoding_key = int(beer_mapping[beer_mapping['beerID']==int(key)]['item_id_idx'].values)
        x_test[encoding_key] = float(value)

    return x_test

def re_transform(topk_pred_list_idx : list):
    beer_mapping = pd.read_csv('data/ratebeer_label_encoding.csv')
    topk_pred_list_item = []
    for idx in topk_pred_list_idx:
        item = beer_mapping[beer_mapping['item_id_idx']==idx]['beerID'].values
        topk_pred_list_item.extend(item)

    return topk_pred_list_item

def predict_from_select_beer(model: AutoRec , data : dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformed_data = _transform(data)
    x_test = torch.tensor(transformed_data).to(device)

    # 추천 맥주 개수
    topk = 4
    # 모델 예측 맥주 평점
    model.eval()
    rating_pred = model(x_test)
    rating_pred = rating_pred.cpu().data.numpy().copy()
    topk_pred_list_idx , topk_rating_list = indexing_from_model(rating_pred, topk)
    topk_pred_list = re_transform(topk_pred_list_idx)
    return topk_pred_list , topk_rating_list

def indexing_from_model(rating_pred : list, topk :int = 4):
    # topk 맥주 index
    ind = np.argpartition(rating_pred, -topk)[-topk:]
    # topk 맥주 index별 평점
    arr_ind = rating_pred[ind]

    # 평점 기준으로 내림차순으로 정렬
    arr_ind_argsort = np.argsort(arr_ind)[::-1]

    # rating 내림차순 모델 예측 topk 맥주 index 
    topk_pred_list = ind[arr_ind_argsort]

    # rating 내림차순 모델 예측 맥주 평점 중 topk개
    topk_rating_list = rating_pred[topk_pred_list]
    return topk_pred_list.tolist() , topk_rating_list.tolist()