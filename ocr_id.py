# -*- coding:utf-8 -*-
import os
import pytesseract
from PIL import Image
import cv2
from common import *
import include.functions as func
import shutil
import include.findidcard as findidcard

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
    img = idfind.find(pic_path)
    if img != False:
        img = img.get()
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        content = pytesseract.image_to_string(img, lang='chi_sim')
    return content

if __name__ == "__main__":
    # pic_path = r"ori_id\test3_2.jpg"
    # print(identity_ocr_chinese(pic_path))
    for item in os.listdir(post_idadd):
        os.remove(post_idadd + item)
    filename = ""
    lastfilename = ""
    filecnt = 0
    blockcnt = 0
    for index, item in enumerate(os.listdir(ori_idadd)):
        if blockcnt >= 2:
            print(lastfilename)
            filename = ""
            lastfilename = ""
            filecnt = 0
            blockcnt = 0
        if filecnt >= 2:
            filename = ""
            lastfilename = ""
            filecnt = 0
            blockcnt = 0
        if item.endswith('.jpg') or item.endswith('.png'):
            pic_path = ori_idadd + item
            content = identity_ocr_chinese(pic_path)
            content = content.replace("\n", "")
            conlist = content.split(" ")
            for con in conlist:
                if func.is_identi_number(con) != False:
                    filename = con
                    break
            if filename != "":
                filecnt += 1
                filenameend = item.split(".")[1]
                newfilename = filename + '_' + str(filecnt) + "." + filenameend
                shutil.copy(pic_path, post_idadd + newfilename)
                if lastfilename != "":
                    filecnt += 1
                    filenameend = item.split(".")[1]
                    newfilename = filename + '_' + str(filecnt) + "." + filenameend
                    shutil.copy(lastfilename, post_idadd + newfilename)
            else:
                lastfilename = pic_path
                blockcnt += 1


