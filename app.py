'''
Author: Gao Liangcai, Wei Baole, Huxingjian 
Date: 2023-1-16 22:18:34
'''
# -*- coding: utf-8 -*-

import json

import cv2
# Flask
from flask import Flask, request, render_template, jsonify
# Some utilites
from gevent.pywsgi import WSGIServer

from pku_ocr_paddle import PKUOCRPP
from util import base64_to_pil, np_to_base64, pil_to_cv2

# Declare a flask app
app = Flask(__name__)

checker = PKUOCRPP(use_gpu=True)
#det_model_path = './work_dirs/release_models/fcenet_r50dcnv2_fpn_1500e_ctw1500_20211022-e326d7ec.pth'
#rec_model_path = './work_dirs/release_models/epoch_20.pth'
#checker = PKUOCR(det='FCE_CTW_DCNv2', recog='SAR_CN', det_ckpt=det_model_path, recog_ckpt=rec_model_path)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #print(request.method)
    if request.method == 'POST':
        # Get the image from post request
        request_json = request.json
        if isinstance(request_json, str):
            request_json = json.loads(request_json)

        if isinstance(request_json['image'], list):
            # batch inference (api interface): img(list[ndarray])
            img = []
            for i in request_json['image']:
                img.append(pil_to_cv2(base64_to_pil(i)))
            if not request_json.get("return_image", False):
                # when not return visualization result
                res_dicts = checker.readtext(img, details=True, return_image=False)
            else:   
                # vis_img detect result img concat recog result img
                res_dicts, vis_img = checker.readtext(img, details=True, return_image=False)
                vis_img = [cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in vis_img]
            
            # convert outputs to json
            outputs = []
            for res_dict in res_dicts:
                opt_str = ''
                if 'score' in res_dict:
                    opt_str += 'text：{}\nscore：{:.3f}'.format(res_dict['text'], res_dict['score'])
                else:
                    for bbox in res_dict['result']:
                        opt_str += 'text：{}\nscore：{:.3f}\n\n'.format(bbox['text'], bbox['text_score'])
                outputs.append(opt_str)
            if not request_json.get("return_image", False):
                return jsonify(format=res_dicts, output=outputs)
            else:
                return jsonify(format=res_dicts, output=outputs, res_image=[np_to_base64(v) for v in vis_img])
        else:
            # single batch inference (especially for web visualization): img is ndarray
            img = base64_to_pil(request_json['image'])
            img = pil_to_cv2(img)
            res_dict, vis_img = checker.readtext(img, details=True)
            res_dict = res_dict[0]
            vis_img = vis_img[0]
            # detect result img concat recog result img
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
            # convert outputs to json
            opt_str = ''
            if 'score' in res_dict:
                opt_str += 'text：{}\nscore：{:.3f}'.format(res_dict['text'], res_dict['score'])
            else:
                for bbox in res_dict['result']:
                    opt_str += 'text：{}\nscore：{:.3f}\n\n'.format(bbox['text'], bbox['text_score'])
            return jsonify(res_dict=dict(format=res_dict, output=opt_str), res_image=np_to_base64(vis_img))

    return None


if __name__ == '__main__':
    # app.run(port=5000, threaded=False)
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
