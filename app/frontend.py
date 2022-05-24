import streamlit as st

# 맥주 id-name dict
beer_dict = {'729':'Sapporo Premium Beer / Draft Beer', '567803':'Tsingtao Premium Stout 4.8%',
'64518':'Tsingtao Draft Beer 11º (Pure Draft Beer)', '37':'Heineken',
'34662':'Heineken Dark Lager', '48076':'Heineken Premium Light'}

beer_list = ['Sapporo Premium Beer / Draft Beer', 'Tsingtao Premium Stout 4.8%', 'Tsingtao Draft Beer 11º (Pure Draft Beer)',
'Heineken', 'Heineken Dark Lager', 'Heineken Premium Light']

beer_img_link = ['https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_729',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_567803',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_64518',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_37',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_34662',
'https://res.cloudinary.com/ratebeer/image/upload/d_beer_img_default.png,f_auto/beer_48076']

st.title("도전맥주홀릭")

# 이전 버전에서 사용 가능
# grid = st.grid()

# for image_info in beer_img_link:
#   row = grid.row()
#   row.image(image_info.image)

options = ["좋아요😊", "몰라요🙄", "싫어요😠"]

col1, col2, col3, col4 = st.columns(4)
col1.image(beer_img_link[0])
col1.write(st.radio(
    "해당 맥주에 대해 어떻게 생각하시나요?!", options
))
col2.write("this is col2")
col3.write("this is col3!!!")
col4.write("this is col4~~~")

# submit 버튼 어떤 것으로 해야할지 search 필요
st.button('제출')

# with st.form(key="제출 form"):
#     st.form_submit_button("제출")