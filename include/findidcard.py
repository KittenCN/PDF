# -*- coding: utf-8 -*-
import numpy as np
import cv2
from PIL import Image
import pytesseract
import re

class findidcard:
    def __init__(self):
        pass

    #img1为身份证模板, img2为需要识别的图像
    def find(self, img2_name, img1_name = r'mask\\idcard_mask.jpg'):
        # print(u'进入身份证模版匹配流程...')
        # img1_name = 'idcard_mask.jpg'
        MIN_MATCH_COUNT = 30
        img1 = cv2.UMat(cv2.imread(img1_name, 0)) # queryImage in Gray
        img1 = self.img_resize(img1, 640)
        # self.showimg(img1)
        #img1 = idocr.hist_equal(img1)
        img2 = cv2.UMat(cv2.imread(img2_name, 0))  # trainImage in Gray
        # print(img2.get().shape)
        img2 = self.img_resize(img2, 1920)
        #img2 = idocr.hist_equal(img2)
        img_org = cv2.UMat(cv2.imread(img2_name))
        img_org = self.img_resize(img_org, 1920)
        #  Initiate SIFT detector
        # t1 = round(time.time() * 1000)

        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 10)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        #两个最佳匹配之间距离需要大于ratio 0.7,距离过于相似可能是噪声点
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        #reshape为(x,y)数组
        if len(good)>=MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            #用HomoGraphy计算图像与图像之间映射关系, M为转换矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            #使用转换矩阵M计算出img1在img2的对应形状
            h,w = cv2.UMat.get(img1).shape
            M_r=np.linalg.inv(M)
            im_r = cv2.warpPerspective(img_org, M_r, (w,h))
            # self.showimg(im_r)
        else:
            # print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            # matchesMask = None
            return False, None

        #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #           singlePointColor = None,
        #           matchesMask = matchesMask, # draw only inliers
        #           flags = 2)
        #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        #plt.imshow(img3, 'gray'),plt.show()
        # t2 = round(time.time() * 1000)
        # print(u'查找身份证耗时:%s' % (t2 - t1))

        img_data_gray, img_org = self.img_resize_gray(im_r)
        im_r = self.find_idnum(img_data_gray, img_org)
        idnum, birth = self.get_idnum_and_birth(im_r)
        return im_r, idnum


    def showimg(self, img):
        cv2.namedWindow("contours", 0);
        #cv2.resizeWindow("contours", 1600, 1200);
        cv2.imshow("contours", img)
        cv2.waitKey()

    def img_resize(self, imggray, dwidth):
        # print 'dwidth:%s' % dwidth
        crop = imggray
        size = crop.get().shape
        height = size[0]
        width = size[1]
        height = height * dwidth / width
        crop = cv2.resize(src=crop, dsize=(dwidth, int(height)), interpolation=cv2.INTER_CUBIC)
        return crop

    x = 1280.00 / 3840.00
    pixel_x = int(x * 3840)

    def find_idnum(self, crop_gray, crop_org):
        template = cv2.UMat(cv2.imread('mask\\idnum_mask_%s.jpg'%self.pixel_x, 0))
        # showimg(template)
        #showimg(crop_gray)
        w, h = cv2.UMat.get(template).shape[::-1]
        res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (max_loc[0] + w, max_loc[1] - int(20*self.x))
        bottom_right = (top_left[0] + int(2300*self.x), top_left[1] + int(300*self.x))
        result = cv2.UMat.get(crop_org)[top_left[1]-10:bottom_right[1], top_left[0]-10:bottom_right[0]]
        cv2.rectangle(crop_gray, top_left, bottom_right, 255, 2)
        #showimg(crop_gray)
        return cv2.UMat(result)

    def img_resize_gray(self, imgorg):     
        #imgorg = cv2.imread(imgname)
        crop = imgorg
        size = cv2.UMat.get(crop).shape
        # print size
        height = size[0]
        width = size[1]
        # 参数是根据3840调的
        height = int(height * 3840 * self.x / width)
        # print height
        crop = cv2.resize(src=crop, dsize=(int(3840 * self.x), height), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return self.hist_equal(img), crop

    def hist_equal(self, img):
            # clahe_size = 8
            # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(clahe_size, clahe_size))
            # result = clahe.apply(img)
            #test

            #result = cv2.equalizeHist(img)

            image = img.get() #UMat to Mat
            # result = cv2.equalizeHist(image)
            lut = np.zeros(256, dtype = image.dtype )#创建空的查找表
            #lut = np.zeros(256)
            hist= cv2.calcHist([image], #计算图像的直方图
                                [0], #使用的通道
                                None, #没有使用mask
                            [256], #it is a 1D histogram
                            [0,256])
            minBinNo, maxBinNo = 0, 255
            #计算从左起第一个不为0的直方图柱的位置
            for binNo, binValue in enumerate(hist):
                if binValue != 0:
                    minBinNo = binNo
                    break
            #计算从右起第一个不为0的直方图柱的位置
            for binNo, binValue in enumerate(reversed(hist)):
                if binValue != 0:
                    maxBinNo = 255-binNo
                    break
            #print minBinNo, maxBinNo
            #生成查找表
            for i,v in enumerate(lut):
                if i < minBinNo:
                    lut[i] = 0
                elif i > maxBinNo:
                    lut[i] = 255
                else:
                    lut[i] = int(255.0*(i-minBinNo)/(maxBinNo-minBinNo)+0.5)
            #计算,调用OpenCV cv2.LUT函数,参数 image --  输入图像，lut -- 查找表
            #print lut
            result = cv2.LUT(image, lut)
            #print type(result)
            #showimg(result)
            return cv2.UMat(result)
    
    def get_idnum_and_birth(self, img):
        _, _, red = cv2.split(img)
        # print('idnum')
        red = cv2.UMat(red)
        red = self.hist_equal(red)
        red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
        red = self.img_resize(red, 150)
        # cv2.imwrite('idnum_red.png', red)
        #idnum_str = get_result_fix_length(red, 18, 'idnum', '-psm 8')
        # idnum_str = get_result_fix_length(red, 18, 'eng', '--psm 8 ')
        img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
        # img.show()
        idnum_str = self.get_result_vary_length(red, 'eng', img, '--psm 8 ')
        return idnum_str, idnum_str[6:14]

    def get_result_vary_length(self, red, langset, org_img, custom_config=''):
        red_org = red
        # cv2.fastNlMeansDenoising(red, red, 4, 7, 35)
        rec, red = cv2.threshold(red, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        # 描边一次可以减少噪点
        cv2.drawContours(red, contours, -1, (255, 255, 255), 1)
        color_img = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)
        numset_contours = []
        height_list=[]
        width_list=[]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            height_list.append(h)
            # print(h,w)
            width_list.append(w)
        height_list.remove(max(height_list))
        width_list.remove(max(width_list))
        height_threshold = 0.70*max(height_list)
        width_threshold = 1.4 * max(width_list)
        # print('height_threshold:'+str(height_threshold)+'width_threshold:'+str(width_threshold))
        big_rect=[]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > height_threshold and w < width_threshold:
                # print(h,w)
                numset_contours.append((x, y, w, h))
                big_rect.append((x, y))
                big_rect.append((x + w, y + h))
        big_rect_nparray = np.array(big_rect, ndmin=3)
        x, y, w, h = cv2.boundingRect(big_rect_nparray)

        # imgrect = cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # self.showimg(imgrect)
        # # self.showimg(cv2.UMat.get(org_img)[y:y + h, x:x + w])
        # self.showimg(cv2.UMat.get(red_org))

        result_string = ''
        result_string += pytesseract.image_to_string(cv2.UMat.get(red_org), lang=langset,
                                                    config=custom_config)
        # print(result_string)
        # cv2.imwrite('varylength.png', cv2.UMat.get(org_img)[y:y + h, x:x + w])
        # cv2.imwrite('varylengthred.png', cv2.UMat.get(red_org)[y:y + h, x:x + w])
        # numset_contours.sort(key=lambda num: num[0])
        # for x, y, w, h in numset_contours:
        #     result_string += pytesseract.image_to_string(cv2.UMat.get(color_img)[y:y + h, x:x + w], lang=langset, config=custom_config)
        return self.punc_filter(result_string)

    def punc_filter(self, str):
        temp = str
        xx      =   u"([\u4e00-\u9fff0-9A-Z]+)"
        pattern =   re.compile(xx)
        results =   pattern.findall(temp)
        string = ""
        for result in results:
            string += result
        return string

if __name__=="__main__":
    idfind = findidcard()
    result = idfind.find(r'ori_id\\test3_2.jpg', r'mask\\idcard_mask.jpg')
    idfind.showimg(result)