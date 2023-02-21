# -*- coding:utf-8 -*-
import os
import pytesseract
from PIL import Image
import cv2
from common import *
import include.functions as func
import shutil
import include.findidcard as findidcard
from tqdm import tqdm

def binarizing(img, threshold):
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img

def depoint(img):
    pixdata = img.load()
    w, h = img.size
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            count = 0
            if pixdata[x, y - 1] > 245:
                count = count + 1
            if pixdata[x, y + 1] > 245:
                count = count + 1
            if pixdata[x - 1, y] > 245:
                count = count + 1
            if pixdata[x + 1, y] > 245:
                count = count + 1
            if count > 2:
                pixdata[x, y] = 255
    return img

def identity_ocr(pic_path):
    image = Image.open(pic_path)
    w, h = image.size
    out = image.resize((int(w * 3), int(h * 3)), Image.ANTIALIAS)
    region = (125 * 3, 200 * 3, 370 * 3, 250 * 3)
    cropImg = out.crop(region)
    img = cropImg.convert('L')
    img = binarizing(img, 100)
    img = depoint(img)
    code = pytesseract.image_to_string(img)
    return code

def identity_ocr_nopro(pic_path):
    img = Image.open(pic_path)
    content = pytesseract.image_to_string(img)
    return content

def identity_ocr_chinese(pic_path):
    # img = Image.open(pic_path)
    content = ""
    idfind = findidcard.findidcard()
    img, idnum = idfind.find(pic_path)
    if img != False:
        img = img.get()
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        content = pytesseract.image_to_string(img, lang='chi_sim')
    return content, idnum

def ocr_id():
    # pic_path = r"ori_id\img (20).jpg"
    # print(identity_ocr_chinese(pic_path))
    endstring = ["jpg", "png"]
    for item in os.listdir(post_idadd):
        os.remove(post_idadd + item)
    filename = [""] * 2
    address = [_ for _ in os.listdir(ori_idadd) if _.split('.')[1] in endstring]
    address.sort(key = lambda x:int(x.split('(')[1].split(')')[0]))
    pbar = tqdm(total=len(address) * len(endstring))
    err_list = []
    for end in endstring:
        index = -1
        for item in address:
            if item.endswith(end):
                index += 1
                filename[index % 2] = ori_idadd + item
                if index % 2 == 1:
                    fn = ""
                    _index = 1
                    for i, pic_path in enumerate(filename):
                        content, idnum = identity_ocr_chinese(pic_path)
                        content = content.replace("\n", "")
                        conlist = content.split(" ")
                
                        if func.is_identi_number(idnum) != False:
                            fn = idnum
                            _index = i
                        else:
                            for con in conlist:
                                if func.is_identi_number(con) != False:
                                    fn = con
                                    _index = i
                                    break
                        if fn != "":
                            break

                    if fn != "":
                        filenameend = item.split(".")[1]
                        newfilename = fn + '_2' + "." + filenameend
                        shutil.copy(filename[_index], post_idadd + newfilename)
                        newfilename = fn + '_1' + "." + filenameend
                        if _index == 1:
                            _index = 0
                        else:
                            _index = 1
                        shutil.copy(filename[_index], post_idadd + newfilename)
                    else:
                        # print("No id number found in " + filename[0] + " and " + filename[1])
                        err_list.append("No id number found in " + filename[0] + " and " + filename[1])
            pbar.update(1)
    pbar.close()
    for item in err_list:
        print(item)

if __name__ == "__main__":
    ocr_id()