"""
data의 수가 부족하므로, 최대한 중요한 정보만 representation할 수 있도록 전처리한다.

k-means 알고리즘을 통해서, pixel단위로 classification을 진행한다.

"""

import cv2
from os import listdir
from os.path import isfile, join
import os
from matplotlib import pyplot as plt
import numpy as np

folder=['trainA','trainB',"testA","testB"]


for f_name in folder:
    imgList = []
    print(f_name)
    imglist=listdir(join('./datasets/dataset/',f_name))
    for file in imglist:
        if "jpg" in file:
            img = cv2.imread(join('./datasets/dataset/',f_name,file))
            print(img.shape)
            Z = img.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 10
            ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((img.shape))

            # path가 존재하지 않을 경우처리
            if not os.path.exists(join('./datasets/dataset/',f_name+"_pre")):
                os.mkdir(join('./datasets/dataset/',f_name+"_pre"))
                print(join('./datasets/dataset/',f_name+"_pre"))

            cv2.imwrite(join('./datasets/dataset/',f_name+"_pre",file), res2)




