# -*- coding:utf-8 -*-
from PIL import Image
import os
from common import *
from sign import sign
import pandas as pd
from tqdm import tqdm

def create_pdf():
    for item in os.listdir(post_processadd):
        os.remove(post_processadd + item)
    for item in os.listdir(pre_processadd):
        os.remove(pre_processadd + item)
    
    csvfile = pd.read_csv(r'./csv/data.csv').values

    pbar = tqdm(total=len(csvfile))
    for row in csvfile:
        sign(row[0], row[1])
        pbar.update(1)
        pbar.set_description("create temp jpg:")
    pbar.close()

    tempfiles = os.listdir(templateadd)
    prefiles = os.listdir(pre_processadd)
    jpgfiles = []
    for tmp in tempfiles:
        if tmp.endswith('.jpg'):
            jpgfiles.append(templateadd + tmp)
    jpgfiles.sort()
    pbar = tqdm(total=len(prefiles))
    for index, pre in enumerate(prefiles):
        source = []
        if pre.endswith('.jpg'):
            _jpgfiles = jpgfiles.copy()
            _jpgfiles.append(pre_processadd + pre)
            output = Image.open(_jpgfiles[0])
            _jpgfiles.pop(0)
            for jpg in _jpgfiles:
                _jpgfile = Image.open(jpg)
                if _jpgfile.mode == "RGB":
                    _jpgfile = _jpgfile.convert("RGB")
                source.append(_jpgfile)
            _name = pre.split('.')[0].replace(' ', '').strip()
            output.save(post_processadd + str(_name) + ".pdf", "pdf", save_all=True, append_images=source)
        pbar.update(1)
        pbar.set_description("create pdf:")
    pbar.close()

if __name__ == "__main__":
    create_pdf()


                    