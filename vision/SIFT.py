import numpy as np
import cv2
from matplotlib import pyplot as plt
from image_registration_pipeline import Display
# get keypoints
"""
img = cv2.imread('./image/insurance_template.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create() 
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(img,kp,img)
cv2.imwrite('sift_keypoints.jpg',img)
cv2.imwrite('sift_gray.jpg',gray)
"""

MIN_MATCH_COUNT = 10
FEATURE_MATCHING = 0

img1 = cv2.imread('./image/box.png',0)          # queryImage
img2 = cv2.imread('./image/box_in_scene.png',0) # trainImage
# plt.imshow(img1, 'gray'),plt.show()

# Initiate SIFT detector
# sift = cv2.SIFT()
sift = cv2.SIFT_create() 
#find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

print('get the keypoints and descriptors with SIFT =>')
# Display().show_keypoints(kp2)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

print('good matches =>')
print(str(len(good)))
# Display().show_matches(good,mode=FEATURE_MATCHING)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()  # RANSAC所篩選出的inliers陣列，用0,1表示good matches哪些事inlier/outliers
    print('get matchesMask type => '+str(type(matchesMask)))
    print(str(matchesMask))
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0],[0,h/2] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    print('get dst => '+str(dst.shape))
    print(str(dst))
    img2 = cv2.polylines(img2,[np.int32(dst)],True,80,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imwrite('SIFT_matches.png',img3)
plt.imshow(img3, 'gray'),plt.show()