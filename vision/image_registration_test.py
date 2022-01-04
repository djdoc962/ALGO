from image_registration_pipeline import Data, Display, PipelineBase, LoadImage, LocalFeaturesPairs, FeatureExtraction, FeatureMatching, ImageAlignment, Evaluation
from typing import Dict, List, Any, Union, Tuple, get_type_hints


MAX_FEATURES = 100
# ORB: 0,
FEATURE_EXTRACTION = 0 
# BRUTEFORCE_HAMMING: 0, FLANN: 1
FEATURE_MATCHING = 0   
MATCHES_PERCENT = 0.5
MATCHES_FILTER = True
BASE_PATH = './outcome/'
TEMPLATE_PATH = BASE_PATH + 'template_keypoints_' + str(FEATURE_MATCHING) + '_' + str(MAX_FEATURES) + '_' + str(MATCHES_PERCENT) + '.png'
QUERY_PATH = BASE_PATH + 'query_keypoints_' + str(FEATURE_MATCHING) + '_' + str(MAX_FEATURES) + '_' + str(MATCHES_PERCENT) + '.png'
MATCHES_PATH = BASE_PATH + 'matches_' +str(FEATURE_MATCHING) + '_' + str(MAX_FEATURES) + '_' + str(MATCHES_PERCENT) + '.png'

print('[START] Experiment Pipeline testing ------------------------------')
print('- Specified data list: ')
print(get_type_hints(Data))
## template image 
template_data = Data('template_data',[('img_path','./image/box_in_scene.png'),('input_image',None),('input_keypoints',None),('input_descriptors',None)]).get_data()
processes = [LoadImage,FeatureExtraction(maxFeatures=MAX_FEATURES,method=FEATURE_EXTRACTION)]
vision_pipeline = PipelineBase(processes)
template_data = vision_pipeline.execute(template_data)
Display().draw_keypoints(template_data.input_image,template_data.input_keypoints,save_path=TEMPLATE_PATH)

## query image 
query_data = Data('query_data',[('img_path','./image/box.png'),('input_image',None),('input_keypoints',None),('input_descriptors',None)]).get_data()
# print('_LoadImage_FeatureExtraction => '+str(query_data)) 
processes = [LoadImage,FeatureExtraction(maxFeatures=MAX_FEATURES,method=FEATURE_EXTRACTION)]
vision_pipeline = PipelineBase(processes)
query_data = vision_pipeline.execute(query_data)
Display().show_keypoints(query_data.input_keypoints)
Display().show_descriptors(query_data.input_descriptors)

#TODO draw keypoint on image
Display().draw_keypoints(query_data.input_image,query_data.input_keypoints,save_path=QUERY_PATH)

matches_data = Data('matches_data',[('query_image',query_data.input_image),('query_descriptors',query_data.input_descriptors),('query_keypoints',query_data.input_keypoints),('template_image',template_data.input_image),('template_descriptors',template_data.input_descriptors),('template_keypoints',template_data.input_keypoints)]).get_data()
vision_pipeline = PipelineBase([FeatureMatching(keepPercent=MATCHES_PERCENT,filter=MATCHES_FILTER, method=FEATURE_MATCHING),ImageAlignment])
matches_data = vision_pipeline.execute(matches_data)
print('matchesMask => '+ str(matches_data.matchesMask))
Display().show_matches(matches_data.matches,mode=FEATURE_MATCHING)
Display().draw_matches(query_data.input_image, query_data.input_keypoints, template_data.input_image, template_data.input_keypoints, matches_data.matches, mode=FEATURE_MATCHING,matchesMask=None, save_path=MATCHES_PATH)

evaluation_data = Data('evaluation_data',[('template_image',template_data.input_image),('template_keypoints',template_data.input_keypoints),('query_keypoints',query_data.input_keypoints),('homography',matches_data.homography)]).get_data()
vision_pipeline = PipelineBase([Evaluation])
evaluation_data = vision_pipeline.execute(evaluation_data)