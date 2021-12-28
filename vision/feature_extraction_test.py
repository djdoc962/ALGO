import os,datetime
import cv2
from feature_extraction import FeatureExtraction, ImageAlignment, Display

img_path = "./image/insurance_query.jpg"
template_path = "./image/insurance_template.jpg"
maxFeatures = 200
keepPercent = 0.5
print('[START] image alignment is started ')
input_img = cv2.imread(img_path)
print('type of input_img '+ str(type(input_img)))
# gray_img = cv2.imread(img_path,0)
template_img = cv2.imread(template_path)
(h, w) = template_img.shape[:2]
print('Template image (heigh, width): '+str(h)+', '+str(w))
(h, w) = input_img.shape[:2]
print('Input image (heigh, width): '+str(h)+', '+str(w))

tic = datetime.datetime.now()
register = ImageAlignment()

(keypointsA, descripsA) = FeatureExtraction().get_keypoint(input_img,maxFeatures=maxFeatures)
print('type of keypointsA '+ str(type(keypointsA)))
Display().show_keypoints(keypointsA)
print('type of descripsA '+ str(type(descripsA)))
Display().show_descriptors(descripsA)
(keypointsB, descripsB) = FeatureExtraction().get_keypoint(template_img,maxFeatures=maxFeatures)
print("preprocess: The processing of keypoints detection takes elapsed time : ", datetime.datetime.now() - tic)

tic = datetime.datetime.now()
matches = register.match(descripsA, descripsB, keepPercent=keepPercent)
print('type of matches '+ str(type(matches)))
Display().show_matches(matches)
print("preprocess: The processing of descriptors matching takes elapsed time : ", datetime.datetime.now() - tic)

tic = datetime.datetime.now()
H_matrix = register.find_homography(keypointsA, keypointsB, matches)
print('type of H_matrix '+ str(type(H_matrix)))
print("preprocess: The processing of homography matrix calculation takes elapsed time : ", datetime.datetime.now() - tic)

tic = datetime.datetime.now()
aligned_img = register.wrap(input_img, H_matrix, template_img.shape[:2] )
print('type of aligned_img '+ str(type(aligned_img)))
cv2.imwrite('./'+os.path.basename(img_path[:-4])+'_align_'+str(maxFeatures)+'_'+str(keepPercent)+'.png',aligned_img)
print('[FINISH] image alignment is finished ')
print("preprocess: The processing of image wraping takes elapsed time : ", datetime.datetime.now() - tic)
