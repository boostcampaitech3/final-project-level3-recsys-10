from requests import request
import streamlit as st
import requests

from utils import resize_image, save_rate
from config import CGF

#################################
# 맥주 DB
# TODO DB를 연동해 주세요

beer_dict = {'Sapporo Premium Beer / Draft Beer':'729', 'Tsingtao Premium Stout 4.8%':'567803',
'Tsingtao Draft Beer 11º (Pure Draft Beer)':'64518', 'Heineken':'37',
'Heineken Dark Lager':'34662', 'Heineken Premium Light':'48076'}

beer_list = ['Sapporo Premium Beer / Draft Beer', 'Tsingtao Premium Stout 4.8%', 'Tsingtao Draft Beer 11º (Pure Draft Beer)',
'Heineken', 'Heineken Dark Lager', 'Heineken Premium Light', 'Heineken Premium Light', 'Heineken Premium Light']

beer_img_link = ['https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_729',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_567803',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_64518',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_37',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_34662',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_48076',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_48076',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_48076']

#################################
# 저장변수
coldstart_data = {}

##################################
# streamlit 실행
st.title("도전맥주홀릭")

# coldstart 맥주 평가
st.header("맥주에 대해 평가해 주세요")

col1, col2, col3, col4 = st.columns(4) # streamlit은 grid 방식을 지원하지 않기 때문에 columns로 hard coding
col5, col6, col7, col8 = st.columns(4)
cols = [col1, col2, col3, col4, col5, col6, col7, col8] 

for col, beer, image_path in zip(cols,beer_list,beer_img_link):
    with col:
        st.image(resize_image(image_path, CGF.image_size))
        option_rate = st.radio(
            beer, CGF.options, key = col
        )
        save_rate(beer_dict, beer, option_rate, coldstart_data)


# coldstart 맥주 평가 제출
coldstart_button = st.button('제출')
if coldstart_button: 
    st.write(coldstart_data)
    # TODO 여기서 POST를 해주세요
    st.write(type(coldstart_data))
    response = requests.post("http://localhost:8001/select", json=coldstart_data)
    st.write(response.json())
    st.write(type(response.json()))

# 4캔 추천 config



# 4캔 추천 반환



# 추천 결과 평가