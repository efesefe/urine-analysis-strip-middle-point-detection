import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob as glob

# https://www.javatpoint.com/object-recognition-using-python
# https://towardsdatascience.com/how-to-build-a-weapon-detection-system-using-keras-and-opencv-67b19234e3dd

def detect():
    paths = glob.glob('samples/*/*.png',recursive=True)
    img2 = cv.imread('samples/reference_scale.png',0)
    for i in range(0,len(paths)):
        img1 = cv.imread(paths[i],0)
        imtemp = cv.imread(paths[i])
        MIN_MATCH_COUNT = 10
        # img1 = cv.imread('box.png',0)          # queryImage
        # img2 = cv.imread('box_in_scene.png',0) # trainImage
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h = img1.shape[0]
            w = img1.shape[1]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
        img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        # cv.imshow('gray ', img3)
        # cv.waitKey(0)
        
        # img = cv.imread('rectangle.jpg')
        # imtemp = cv.GaussianBlur(imtemp,(3,3),0)
        # imtemp = adjust_gamma(imtemp, 1.7)
        # imtemp= cv.fastNlMeansDenoisingColored(imtemp,None, 10, 10, 7, 21)
        imtemp = cv.rectangle(imtemp, (10,223), (630,274), (0,0,255), 2)
        cv.imshow('thsresh', imtemp)
        cv.waitKey(0)
        gray_img = cv.cvtColor(imtemp[223:274,18:620], cv.COLOR_BGR2GRAY)

        # thresh_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        # thresh_img = cv.adaptiveThreshold(gray_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

        _, thresh_img = cv.threshold(gray_img,127,255,cv.THRESH_BINARY)
        # cv.imshow('thresh', thresh_img)
        # cv.waitKey(0)
        cnts = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # t = (223,18)
        # cnts = np.add(cnts,[[t]])
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # for i in cnts:
        #     print(i)
        cv.drawContours(imtemp, cnts,-1,(0,255,0),3)
        # cv.imshow('drawn', imtemp)
        # print(cnts)
        for cnt in cnts:
            # print(cnt[0][0][0])
            epsilon = 0.01 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)
            # print(approx[0][0])
            # x_start = cnt[0][0][0]
            # y_start = cnt[0][0][1]
            if len(approx) == 4:
                # imtemp = cv.rectangle(imtemp, (approx[0][0][0],approx[0][0][1]), (approx[0][len(approx[0])-1]), (0,0,255), 2)
                cv.drawContours(imtemp, cnt,-1,(0,255,0),3)
        # cv.imshow('drawn', imtemp)
        # cv.waitKey(0)

        # cv.imshow('strip', img)
        # cv.waitKey(0)


def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	return cv.LUT(image, table)

if __name__ == '__main__':
    detect()
