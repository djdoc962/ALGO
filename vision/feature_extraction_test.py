import os,datetime
import cv2
from feature_extraction import ImageAlignment

img_path = "./image/table4.jpg"
template_path = "./image/template.jpg"
maxFeatures = 200
keepPercent = 0.5
print('[START] image alignment is started ')
input_img = cv2.imread(img_path)
# gray_img = cv2.imread(img_path,0)
template_img = cv2.imread(template_path)
(h, w) = template_img.shape[:2]
print('Template image (heigh, width): '+str(h)+', '+str(w))
(h, w) = input_img.shape[:2]
print('Input image (heigh, width): '+str(h)+', '+str(w))

tic = datetime.datetime.now()
detector = ImageAlignment()
(keypointsA, descripsA) = detector.get_keypoint(input_img,maxFeatures=maxFeatures)
detector.show_keypoints(keypointsA)
detector.show_descriptors(descripsA)
(keypointsB, descripsB) = detector.get_keypoint(template_img,maxFeatures=maxFeatures)
print("preprocess: The processing of keypoints detection takes elapsed time : ", datetime.datetime.now() - tic)

tic = datetime.datetime.now()
matches = detector.match(descripsA, descripsB, keepPercent=keepPercent)
detector.show_matches(matches)
print("preprocess: The processing of descriptors matching takes elapsed time : ", datetime.datetime.now() - tic)

tic = datetime.datetime.now()
H_matrix = detector.find_homography(keypointsA, keypointsB, matches)
print("preprocess: The processing of homography matrix calculation takes elapsed time : ", datetime.datetime.now() - tic)

tic = datetime.datetime.now()
aligned_img = detector.wrap(input_img, H_matrix, template_img.shape[:2] )
# cv2.imwrite('./'+os.path.basename(img_path[:-4])+'_align_'+str(maxFeatures)+'_'+str(keepPercent)+'.png',aligned_img)
print('[FINISH] image alignment is finished ')
print("preprocess: The processing of image wraping takes elapsed time : ", datetime.datetime.now() - tic)
