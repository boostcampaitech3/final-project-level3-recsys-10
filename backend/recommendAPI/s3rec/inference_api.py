import argparse
import os
from typing import Dict
import pandas as pd

import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np

from backend.recommendAPI.s3rec.models import S3RecModel

from backend.recommendAPI.s3rec.utils import (
    check_path,
    get_item2attribute_json,
    set_seed,
)

def inference(input: Dict, filter_ids):
    print("!!!!!!!!!!!!!!!!!!S3Rec!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/train/", type=str)
    parser.add_argument("--output_dir", default="backend/recommendAPI/s3rec/output/", type=str)
    parser.add_argument("--data_name", default="rb", type=str)
    parser.add_argument("--do_eval", action="store_true")

    # model args
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=3, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=300, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=512, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = os.path.join(args.data_dir, args.data_name + "_item2attributes.json")
    # item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"
    max_item = 9335 # (시스템에서 9337일때와 같음) # TODO DB에서 참조하도록 만들기

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    le = LabelEncoder()
    label_path = os.path.join(args.output_dir, "item" + "_classes.npy") # args.asset_dir -> args.output_dir
    le.classes_ = np.load(label_path)

    # save model args
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # model load
    args_str = f"{args.model_name}-{args.data_name}"
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    model = S3RecModel(args=args)
    file_name = args.checkpoint_path
    model.load_state_dict(torch.load(file_name))
    model = model.to(device="cuda:0")

    # input
    beer_pick = list(input.keys())
    input_ids = le.transform(beer_pick).tolist()
    input_ratings = list(input.values()) # TODO

    # 전처리 (padding)
    pad_len = args.max_seq_length - len(input_ids)
    input_ids = [0] * pad_len + input_ids
    input_ratings = [0] * pad_len + input_ratings

    input_ids = input_ids[-args.max_seq_length :]
    input_ratings = input_ratings[-args.max_seq_length :]

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to("cuda:0")
    input_ratings = torch.tensor(input_ratings, dtype=torch.float32).unsqueeze(0).to("cuda:0")

    # 모델 run finetune = inference
    recommend_output = model.finetune(input_ids, input_ratings)
    recommend_output = recommend_output[:, -1, :] 

    result_scores = torch.matmul(model.item_embeddings.weight, recommend_output.transpose(0, 1)).squeeze(1)
    result_scores = result_scores.cpu().data.numpy().copy()
    result_scores = result_scores[1:]
    
    # 점수를 sorting
    sorted_items = result_scores.argsort()

    # 방금 체크 했던거는 제거
    checked_right_before = input_ids[input_ids > 0]
    sorted_items = sorted_items[~pd.Series(sorted_items).isin(checked_right_before)]

    # 원래의 id로 되돌아오기
    sorted_items = le.inverse_transform(sorted_items.argsort()) # 뒤로갈수록 추천해주고 싶은 맥주 

    # 한국 맥주 의 아이디 가져오기
    target_items = [int(*i) for i in filter_ids] # DB불러온 것으로 교체
    sorted_target_items = sorted_items[pd.Series(sorted_items).isin(target_items)]

    K=4
    sorted_target_items_topk = sorted_target_items[-K:]

    return sorted_target_items_topk