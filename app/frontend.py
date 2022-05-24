import sys

import streamlit as st
import urllib.request
from io import BytesIO
from PIL import Image

#################################
# 맥주 DB

# 맥주 id-name dict
beer_dict = {'729':'Sapporo Premium Beer / Draft Beer', '567803':'Tsingtao Premium Stout 4.8%',
'64518':'Tsingtao Draft Beer 11º (Pure Draft Beer)', '37':'Heineken',
'34662':'Heineken Dark Lager', '48076':'Heineken Premium Light'}

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

##################################

# 변수
options = ["좋아요😊", "몰라요🙄", "싫어요😠"]
image_size = [150,420]

def resize_image(image_path,image_size):
    # 동일한 이미지로 만들기 위한 함수
    req = urllib.request.Request(image_path)
    res = urllib.request.urlopen(req).read()
    image = Image.open(BytesIO(res))
    resized_image = image.resize(image_size)
    return resized_image     


st.title("도전맥주홀릭")

st.header("맥주에 대해 평가해 주세요")

# streamlit은 grid 방식을 지원하지 않기 때문에 columns로 hard coding
col1, col2, col3, col4 = st.columns(4)
col5, col6, col7, col8 = st.columns(4)
cols = [col1, col2, col3, col4, col5, col6, col7, col8]


for col, beer, image_path in zip(cols,beer_list,beer_img_link):
    with col:
        st.image(resize_image(image_path, image_size))
        st.write(st.radio(
            beer, options, key = col
        ))


# submit 버튼 어떤 것으로 해야할지 search 필요
st.button('제출')

# with st.form(key="제출 form"):
#     st.form_submit_button("제출")