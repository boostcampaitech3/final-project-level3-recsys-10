import urllib.request
from io import BytesIO
from PIL import Image
from config import CGF

# 동일한 이미지 크기로 만들기 위한 함수
def resize_image(image_path,image_size):
    req = urllib.request.Request(image_path)
    res = urllib.request.urlopen(req).read()
    image = Image.open(BytesIO(res))
    resized_image = image.resize(image_size)
    return resized_image     

# rate를 저장해주는 함수
def save_rate(beer_dict, beer, option_rate, coldstart_data):
    rate = transform_rate(option_rate)
    coldstart_data[beer2id(beer_dict, beer)]=rate


def beer2id(beer_dict, beer):
    return beer_dict[beer]

# 
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