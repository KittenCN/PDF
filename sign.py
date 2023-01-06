# -*- coding:utf-8 -*-
from PIL import Image, ImageFont
from handright import Template, handwrite
from common import *
from random import randint
import os

def sign(text, filename):
    ttfcolors = [(0,0,0),(25,25,112),(0,0,205),(160,32,240),(0,0,139)]
    ttffiles = os.listdir(ttfadd)
    ttffile = []
    for ttf in ttffiles:
        if ttf.endswith('.ttf'):
            ttffile.append(ttfadd + ttf)

    ttfindex = randint(0, len(ttffile) - 1)
    ttfsize = randint(75, 120)
    ttfcolorindex = randint(0, len(ttfcolors) - 1)
    template = Template(
        background = Image.open(r'./sign_template/4.jpg'),
        font = ImageFont.truetype(ttffile[ttfindex], ttfsize),
        line_spacing = ttfsize + randint(0, 20),
        fill=ttfcolors[ttfcolorindex],  # 字体颜色，括号内为RGB的值
        left_margin=500 + randint(0, 100),
        top_margin=1400 + randint(0, 100),
        right_margin=50,
        bottom_margin=50,
        word_spacing=15,
        line_spacing_sigma=6,  # 行间距随机扰动
        font_size_sigma=2,  # 字体大小随机扰动
        word_spacing_sigma=1,  # 字间距随机扰动
        end_chars="，。",  # 防止特定字符因排版算法的自动换行而出现在行首
        perturb_x_sigma=4,  # 笔画横向偏移随机扰动
        perturb_y_sigma=4,  # 笔画纵向偏移随机扰动
        perturb_theta_sigma=0.05,  # 笔画旋转偏移随机扰动
    )
    images = handwrite(text, template)
    for i, im in enumerate(images):
        assert isinstance(im, Image.Image)
        im.save("./pre_process/" + str(filename) + ".jpg")