from image_registration_pipeline2 import Data, Display, PipelineBase, LoadImage, WriteImage, LocalFeaturesPairs, FeatureExtraction, FeatureMatching, ImageAlignment, Evaluation
from typing import Dict, List, Any, Union, Tuple, get_type_hints
import os

MAX_FEATURES = 200
# ORB: 0,
FEATURE_EXTRACTION = 0 
# BRUTEFORCE_HAMMING: 0, FLANN: 1
FEATURE_MATCHING = 0   
MATCHES_PERCENT = 0.5
MATCHES_FILTER = True
BASE_PATH = './outcome/'
KPS_TEMPLATE_PATH = BASE_PATH + 'template_keypoints_' + str(FEATURE_MATCHING) + '_' + str(MAX_FEATURES) + '_' + str(MATCHES_PERCENT) + '.png'
KPS_QUERY_PATH = BASE_PATH + 'query_keypoints_' + str(FEATURE_MATCHING) + '_' + str(MAX_FEATURES) + '_' + str(MATCHES_PERCENT) + '.png'
MATCHES_PATH = BASE_PATH + 'matches_' +str(FEATURE_MATCHING) + '_' + str(MAX_FEATURES) + '_' + str(MATCHES_PERCENT) + '.png'
ALIGN_PATH = BASE_PATH + 'alignment_' +str(FEATURE_MATCHING) + '_' + str(MAX_FEATURES) + '_' + str(MATCHES_PERCENT) + '.png'
QUERY_PATH = './outcome/affine/insurance2_scale_0.5.png'
TEMPLATE_PATH = './image/insurance2_template.png'


CHECK_FOLDER = os.path.isdir(BASE_PATH)
if not CHECK_FOLDER:
    os.makedirs(BASE_PATH)


print('[START] Experiment Pipeline testing ------------------------------')
print('- Specified data list: ')
print(get_type_hints(Data))
## template image 
template_data = Data('template_data',[('img_path',TEMPLATE_PATH),('input_image',None),('input_keypoints',None),('input_descriptors',None)]).get_data()
processes = [LoadImage,FeatureExtraction(maxFeatures=MAX_FEATURES,method=FEATURE_EXTRACTION)]
vision_pipeline = PipelineBase(processes)
template_data = vision_pipeline.execute(template_data)
Display().draw_keypoints(template_data.input_image,template_data.input_keypoints,save_path=KPS_TEMPLATE_PATH)

## query image 
query_data = Data('query_data',[('img_path',QUERY_PATH),('input_image',None),('input_keypoints',None),('input_descriptors',None)]).get_data()
# print('_LoadImage_FeatureExtraction => '+str(query_data)) 
processes = [LoadImage,FeatureExtraction(maxFeatures=MAX_FEATURES,method=FEATURE_EXTRACTION)]
vision_pipeline = PipelineBase(processes)
query_data = vision_pipeline.execute(query_data)
# Display().show_keypoints(query_data.input_keypoints)
# Display().show_descriptors(query_data.input_descriptors)

## draw keypoint on image
Display().draw_keypoints(query_data.input_image,query_data.input_keypoints,save_path=KPS_QUERY_PATH)

matches_data = Data('matches_data',[('query_image',query_data.input_image),('query_descriptors',query_data.input_descriptors),('query_keypoints',query_data.input_keypoints),('template_image',template_data.input_image),('template_descriptors',template_data.input_descriptors),('template_keypoints',template_data.input_keypoints)]).get_data()
vision_pipeline = PipelineBase([FeatureMatching(keepPercent=MATCHES_PERCENT,filter=MATCHES_FILTER, method=FEATURE_MATCHING),ImageAlignment])
matches_data = vision_pipeline.execute(matches_data)
print('matchesMask => '+ str(len(matches_data.matchesMask))) ## 跟putative數量一樣
print('all matches => '+ str(len(matches_data.matches)))
print('putative matches => '+ str(len(matches_data.putative_matches)))
Display().show_matches(matches_data.matches,mode=FEATURE_MATCHING)
Display().draw_matches(query_data.input_image, query_data.input_keypoints, template_data.input_image, template_data.input_keypoints, matches_data.putative_matches, mode=FEATURE_MATCHING,matchesMask=None, save_path=MATCHES_PATH)
Display().draw_matches(query_data.input_image, query_data.input_keypoints, template_data.input_image, template_data.input_keypoints, matches_data.putative_matches, mode=FEATURE_MATCHING,matchesMask=matches_data.matchesMask, save_path=MATCHES_PATH[:-4]+'_inliers.png')

WriteImage().execute(matches_data.aligned_image, save_path=ALIGN_PATH)
evaluation_data = Data('evaluation_data',[('template_image',template_data.input_image),('template_keypoints',template_data.input_keypoints),('query_keypoints',query_data.input_keypoints),('homography',matches_data.homography),('matches',matches_data.matches),('putative_matches',matches_data.putative_matches)]).get_data()
vision_pipeline = PipelineBase([Evaluation])
evaluation_data = vision_pipeline.execute(evaluation_data)