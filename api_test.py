'''
Author: Hu Xingjian
Date: 2023-01-16 11:01:26
LastEditTime: 2023-01-16 22:50:13
FilePath: /pkuocr/api_test.py
Description: 
Software:VSCode,env:
'''
import requests
import base64
import json
from PIL import Image
import re
from io import BytesIO

def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image

with open("./test_img.jpg", "rb") as f:
    image = base64.b64encode(f.read()) 
    image = image.decode('utf-8')
js = {'image': [image, image], "return_image":False}
headers = {"Content-Type": "application/json"}

response = requests.post(url='http://10.136.36.29:35552/predict', json=js, headers=headers)
response = response.json()
print(response)
