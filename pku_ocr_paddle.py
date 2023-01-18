'''
Author: Gao Liangcai, Wei Baole, Huxingjian 
Date: 2023-1-16 22:18:26
'''

import os

import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from paddleocr import PaddleOCR

def dist(t1, t2):
    return ((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2)**0.5


def gen_color():
    """Generate BGR color schemes."""
    color_list = [(101, 67, 254), (154, 157, 252), (173, 205, 249),
                  (123, 151, 138), (187, 200, 178), (148, 137, 69),
                  (169, 200, 200), (155, 175, 131), (154, 194, 182),
                  (178, 190, 137), (140, 211, 222), (83, 156, 222)]
    return color_list


def draw_polygons(img, polys):
    """Draw polygons on image.

    Args:
        img (np.ndarray): The original image.
        polys (list[list[float]]): Detected polygons.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    dst_img = img.copy()
    color_list = gen_color()
    out_img = dst_img
    for idx, poly in enumerate(polys):
        poly = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(
            img,
            np.array([poly]),
            -1,
            color_list[idx % len(color_list)],
            thickness=cv2.FILLED)
        out_img = cv2.addWeighted(dst_img, 0.5, img, 0.5, 0)
    return out_img


def draw_texts_by_pil(img,
                      texts,
                      boxes=None,
                      draw_box=True,
                      on_ori_img=False,
                      font_size=None,
                      fill_color=None,
                      draw_pos=None,
                      return_text_size=False,
                      font_path=None):
    """Draw boxes and texts on empty image, especially for Chinese.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else on a new empty image.
        font_size (int, optional): Size to create a font object for a font.
        fill_color (tuple(int), optional): Fill color for text.
        draw_pos (list[tuple(int), tuple(int)], optional): Start point to end point to draw each text.
        return_text_size (bool): If True, return the list of text size.

    Returns:
        (np.ndarray, list[tuple]) or np.ndarray: Return a tuple
        ``(out_img, text_sizes)``, where ``out_img`` is the output image
        with texts drawn on it and ``text_sizes`` are the size of drawing
        texts. If ``return_text_size`` is False, only the output image will be
        returned.
    """

    color_list = gen_color()
    h, w = img.shape[:2]
    if boxes is None:
        boxes = [[0, 0, w, 0, w, h, 0, h]]
    if draw_pos is None:
        draw_pos = [None for _ in texts]
    assert len(boxes) == len(texts) == len(draw_pos)

    if fill_color is None:
        fill_color = (0, 0, 0)

    '''
    if on_ori_img:
        out_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        out_img = Image.new('RGB', (w, h), color=(255, 255, 255))
    out_draw = ImageDraw.Draw(out_img)
    '''

    if on_ori_img:
        res_img = img
    else:
        res_img = np.ones_like(img, dtype=np.uint8) * 255
    

    text_sizes = []
    for idx, (box, text, ori_point) in enumerate(zip(boxes, texts, draw_pos)):
        if len(text) == 0:
            continue
        min_x, max_x = min(box[0::2]), max(box[0::2])
        min_y, max_y = min(box[1::2]), max(box[1::2])
        color = tuple(list(color_list[idx % len(color_list)])[::-1])
        if draw_box:
            out_draw.line(box, fill=color, width=1)
        dirname, _ = os.path.split(os.path.abspath(__file__))
        tmp_font_size = font_size
        if tmp_font_size is None:
            box_width = max(max_x - min_x, max_y - min_y)
            tmp_font_size = int(0.9 * box_width / len(text))
        fnt = ImageFont.truetype(font_path, tmp_font_size)
        if ori_point is None:
            ori_point = (min_x + 1, min_y + 1), (max_x - 1, max_y - 1)
        
        out_img = Image.new('RGB', (h, w), color=(255, 255, 255))
        out_draw = ImageDraw.Draw(out_img)
        out_draw.text((0,0), text, font=fnt, fill=fill_color)
        tw, th = fnt.getsize(text)
        out_img = cv2.cvtColor(np.asarray(out_img)[:th, :tw], cv2.COLOR_RGB2BGR)

        src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
        if dist(box[0:2], box[2:4]) > dist(box[0:2], box[6:]) or len(text)==1:
            dst = np.array(box, dtype=np.float32).reshape(4,2)
        else:
            dst = np.array(box[2:]+box[0:2], dtype=np.float32).reshape(4,2)
        M = cv2.getPerspectiveTransform(src, dst)
        res = cv2.warpPerspective(out_img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        res_img = 255 - cv2.add((255 - res_img), (255 - res))
        text_sizes.append(fnt.getsize(text))

    del out_draw

    if return_text_size:
        return res_img, text_sizes

    return res_img


class PKUOCRPP:
    def __init__(self, **kwargs):
        # need to run only once to download and load model into memory
        self.paddle_engine = PaddleOCR(use_angle_cls=True,
                                       lang='ch', drop_score=0.5,
                                       **kwargs)  
                                

    def readtext(self, img, details=True, return_image=True):
        # default：single image input, return single image recognition result(list)
        if not isinstance(img, list):
            img = [img]
        result = self.paddle_engine.ocr(img, cls=True)
        boxes_ = [[line[0] for line in r]for r in result]
        boxes = []
        for img_box in boxes_:
            img_cache = []
            for box in img_box:
                box_cache = []
                for p in box:
                    box_cache += list(map(float, p))
                img_cache.append(box_cache)
            boxes.append(img_cache)
        txts = [[str(line[1][0]) for line in r] for r in result]
        scores = [[float(line[1][1]) for line in r] for r in result]
        angles = [[int(line[2][0]) for line in r] for r in result]

        batch_size = len(result)

        res_dicts = []
        for n in range(batch_size):
            res_dict = dict()
            res_dict['result'] = []
            for box, txt, score in zip(boxes[n], txts[n], scores[n]):
                bbox = dict()
                if details:
                    bbox['box'] = box
                bbox['text'] = txt
                bbox['text_score'] = score
                res_dict['result'].append(bbox)
            res_dicts.append(res_dict)


        # generate visualization result (especially on web)
        if return_image:
            out_img = []
            for n in range(batch_size):
                box_vis_img = draw_polygons(img[n], boxes[n])
                for i in range(len(boxes)):
                    if angles[i] == 180:
                        boxes[i] = boxes[i][4:] + boxes[i][0:4]
                text_vis_img = draw_texts_by_pil(img[n], txts[n], boxes[n], draw_box=False, font_path='chinese_cht.ttf')
                h, w = img[n].shape[:2]
                vis_img = np.ones((h, w * 2, 3), dtype=np.uint8)        
                vis_img[:, :w, :] = box_vis_img
                vis_img[:, w:, :] = text_vis_img
                out_img.append(vis_img)
            return res_dicts, out_img
        else:
            return res_dicts

def clean_results(result, threshold=0.8):
    '''
        filter result whose confidence lower than threshold
    '''
    res = []
    for line in result:
        text_score = float(line[1][1])       
        if text_score > threshold:
            res.append(line)
    return res