import urllib.request
from io import BytesIO
from PIL import Image
from config import CGF

import streamlit as st
import numpy as np


# 동일한 이미지 크기로 만들기 위한 함수
def resize_image(image_path,image_size):
    req = urllib.request.Request(image_path)
    res = urllib.request.urlopen(req).read()
    image = Image.open(BytesIO(res))
    resized_image = image.resize(image_size)
    return resized_image     


# 
@st.cache
def get_init_beers(beer_list, beer_dict):
    # TODO 보여줄 맥주 알고리즘 
    beer_list = np.random.choice(beer_list, CGF.num_items)
    beer_img_link = [f'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_{beer_dict[beer_name]}' for beer_name in beer_list]
    return beer_list, beer_img_link



# radio-box로 지정한 rate를 저장해주는 함수
def save_rate(beer_dict, beer, option_rate, coldstart_data):
    rate = transform_rate(option_rate)
    coldstart_data[beer2id(beer_dict, beer)]=rate

def beer2id(beer_dict, beer):
    return beer_dict[beer]

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
def get_recommended_beer():  
    # TODO inference 결과를 받아와 주세요 / return 값은 beer_id가 좋을 것 같습니다.
    # 우선 임시로 4개를 받아 옵니다.
    tmp_dict = {'Tsingtao Premium Stout 4.8%':'567803', 'Heineken':'37',
                'Heineken Dark Lager':'34662', 'Heineken Premium Light':'48076'}
    # beer_id를 반환합니다.
    return tmp_dict.values() 


# beer_id로 맥주정보를 받아오는 함수
def get_info(beer_id):
    # TODO beer_id로 필요한 데이터를 조합해주세요
    image_path =  f'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_{beer_id}'
    beer_name = 'Heineken'
    return {'imageUrl' : image_path, 'beerName':beer_name}


# 추천 결과 피드백 제출
def post_feedback(r_beer_id, button):
    if button == "좋아요":
        return r_beer_id, CGF.LIKE
    elif button == "싫어요":
        return r_beer_id, CGF.UNLIKE