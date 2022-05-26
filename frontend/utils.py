import urllib.request
from io import BytesIO
from PIL import Image
from config import CGF

import streamlit as st
import numpy as np
import pandas as pd

# 동일한 이미지 크기로 만들기 위한 함수
def resize_image(image_path,image_size):
    req = urllib.request.Request(image_path)
    res = urllib.request.urlopen(req).read()
    image = Image.open(BytesIO(res))
    resized_image = image.resize(image_size)
    return resized_image     

@st.cache
def get_init_beers(beer_df : pd.DataFrame):
# TODO 보여줄 맥주 알고리즘 
    beer_list = np.random.choice(beer_df['beerName'].values, CGF.num_items, replace=False)
    beer_img_link = []
    for beer_name in beer_list:
        idx = int(beer_df[beer_df['beerName']==beer_name]['beerID'].values)
        img_link = [f'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_{idx}']
        beer_img_link.extend(img_link)

    return beer_list, beer_img_link

# @st.cache
# def get_init_beers(beer_list, beer_dict):
#     # TODO 보여줄 맥주 알고리즘 
#     beer_list = np.random.choice(beer_list, CGF.num_items)
#     beer_img_link = [f'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_{beer_dict[beer_name]}' for beer_name in beer_list]
#     return beer_list, beer_img_link


@st.cache
def get_beer_info():
    #return pd.read_json("data/ratebeer_korea.json").drop_duplicates()
    return pd.read_csv('data/ratebeer_label_encoding.csv')

# radio-box로 지정한 rate를 저장해주는 함수
def save_rate(beer_df : pd.DataFrame, beer : str, option_rate : int, coldstart_data : dict):
    rate = transform_rate(option_rate)
    coldstart_data[beer2id(beer_df, beer)]=rate

def beer2id(beer_df, beer_name):
    return int(beer_df[beer_df['beerName']==beer_name]['beerID'].values)

def transform_rate(option_rate):
    if option_rate == CGF.options[0]:
        rate =  CGF.LIKE
    elif option_rate == CGF.options[1]:
        rate =  CGF.UNKNOWN
    elif option_rate == CGF.options[2]:
        rate =  CGF.UNLIKE
    else:
        rate =  CGF.UNKNOWN
    return rate

# st.columns를 grid 형식으로 하기위한 함수
def get_grid(num_item):
    row = num_item // CGF.max_col
    return row, CGF.max_col

def get_cols(row, col):
    total_cols = []
    for r in range(row):
        cols = []
        for c in range(col):
            globals()[f'col_{r}_{c}'] = 0
        cols = st.columns(col)
        
        total_cols.extend(cols)

    return total_cols


# 맥주추천 결과 받아오는 함수
def get_recommended_beer(response : dict):  
    # TODO inference 결과를 받아와 주세요 / return 값은 beer_id가 좋을 것 같습니다.
    
    # beer_id를 반환합니다.
    return response.keys() 


# beer_id로 맥주정보를 받아오는 함수
def get_info(beer_id, ratebeer):
    # TODO beer_id로 필요한 데이터를 조합해주세요
    # TODO dataframe을 db와 연동되면 빼주세요
    image_path =  f'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_{beer_id}'
    beer_name = ratebeer[ratebeer['beerID'] == int(beer_id)]['beerName'].values[0]
    return {'imageUrl' : image_path, 'beerName':beer_name}


# 추천 결과 피드백 제출
def post_feedback(r_beer_id, button):
    if button == "좋아요":
        return r_beer_id, CGF.LIKE
    elif button == "싫어요":
        return r_beer_id, CGF.UNLIKE