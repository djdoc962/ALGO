import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def Brute_Force(des1,des2):
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    print('bf len: '+str(len(matches)))
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    return matches
  

def drawMatches(matches,query_img,train_img,query_kpt,train_kpt):
    # Draw first 10 matches.
    matches_img = cv.drawMatches(query_img,query_kpt,train_img,train_kpt,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matches_img),plt.show()



if __name__ == "__main__":
    print('Brute-Force matcher method:')
    query_img = cv.imread('./image/box.png',cv.IMREAD_GRAYSCALE)          # queryImage
    train_img = cv.imread('./image/box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(query_img,None)
    kp2, des2 = orb.detectAndCompute(train_img,None)
    print('- get keypoints: len=> {}, {}'.format(len(kp1),len(kp2)))
    print('- get descriptors: len=> {}, {}'.format(len(des1),len(des2)))
    matches = Brute_Force(des1,des2)
    print('- get matches: len=>{} '.format(len(matches)))
    print('- drawing matches ... ')
    drawMatches(matches,query_img,train_img,kp1,kp2)