from typing import Dict, List, Any, Union, Tuple
from dataclasses import dataclass, asdict
import inspect
import cv2
import numpy as np

class Display:
    def show_keypoints(self,keypoints):
        print('keypoints length: {} ======================================'.format(len(keypoints)))
        for i, keypoint in enumerate(keypoints):
            print('i: {}, (x,y): ({}, {})'.format(i,keypoint.pt[0],keypoint.pt[1]))

    def show_descriptors(self,descriptors):
        print('descriptors length: {} with dimension: {} ======================================'.format(len(descriptors),len(descriptors[0])))
        for i, feature in  enumerate(descriptors):
            print('i: {}, feature: {}'.format(i,feature[:]))

    def show_matches(self,matches):
        """
        matches: DMatch object from OpenCV
        match.trainIdx: Index of the descriptor in train descriptors
        match.queryIdx: Index of the descriptor in query descriptors
        match.distance: Distance between descriptors. The lower, the better it is.
        DMatch.imgIdx: Index of the train image
        """
        print('matches length: {} ======================================'.format(len(matches)))
        for i, match in enumerate(matches):
            print('i: {}, Idx: {}, {}'.format(i,match.queryIdx,match.trainIdx))


class FeatureExtraction:
    def __init__(self,
        maxFeatures: int = 200,
        keepPercent: float = 0.5) -> None:
        print('feature_extraction.__init__')
        # Parameters of feature extraction
        self.maxFeatures = maxFeatures
        self.keepPercent = keepPercent
        
    def execute(self, data: Any) -> Any:
        print('FeatureExtraction.execute =>'+str(data))
     
        # print(asdict(data))
        if not Data.check_exists(data.query_keypoints):
            print('[{}] `query_keypoints` is not existed, extracting it now...'.format(self.__class__.__name__))
            (keypoints, descrips) = FeatureExtraction().get_keypoint(data.query_image,maxFeatures=self.maxFeatures)
            data.query_keypoints = keypoints
            data.query_descriptors = descrips
        elif not Data.check_exists(data.template_keypoints):
            print('[{}] `template_keypoints` is not existed, extracting it now...'.format(self.__class__.__name__))
            (keypoints, descrips) = FeatureExtraction().get_keypoint(data.template_image,maxFeatures=self.maxFeatures)
            data.template_keypoints = keypoints
            data.template_descriptors = descrips
        else:
            print('[{}] `query_keypoints/descriptors` and `template_keypoints/descriptors` are already existed. '.format(self.__class__.__name__))
            print('[{}] checking `query_keypoints/descriptors` and `template_keypoints/descriptors` ... '.format(self.__class__.__name__)) 
            Data.check_type(data.query_keypoints,list)
            Data.check_type(data.query_descriptors,np.ndarray)
            Data.check_type(data.template_keypoints,list)
            Data.check_type(data.template_descriptors,np.ndarray)
        

        return data


    def get_keypoint(self, image, maxFeatures=500):
        """
        detect keypoints and extract (binary) local invariant features
        keypoints: FAST keypoint including coordination
        descrips: BRIEF descriptor(32 dimensions By default)
        """
        # convert both the input image to grayscale
        if( len(image.shape) > 2 ):
            print('It is a color image, will be converted to gray image.')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
        orb = cv2.ORB_create(maxFeatures)
        (keypoints, descrips) = orb.detectAndCompute(image, None)
        return keypoints, descrips


class FeatureMatching:
    def __init__(self,keepPercent: float = 0.5) -> None:
        self.keepPercent = keepPercent

    def execute(self, data: Any) -> Any:
        print('FeatureMatching.execute ')
        Data.check_type(data.query_descriptors,np.ndarray)
        Data.check_type(data.template_descriptors,np.ndarray)
            
        matches = ImageAlignment.match(data.query_descriptors,data.template_descriptors, keepPercent=self.keepPercent)
        data.matches = matches
        return data
 

class ImageAlignment:

    @classmethod
    def match(cls, descripA, descripB,keepPercent=0.2, method=0):
        """
        match the features between images
        matches: DMatch object from OpenCV
        """
        if method == 0:
            method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        else:
            method = cv2.DescriptorMatcher_FLANNBASED
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(descripA, descripB, None)
        matches = sorted(matches, key=lambda x:x.distance)
        # keep only the top matches
        keep = int(len(matches) * keepPercent)
        matches = matches[:keep]
        return matches

    @classmethod
    def Match2Keypoint(cls,matches,KptA,KptB):
        """
        get matching keypoints for each of the images
        ptsA: coordinates of image A
        ptsB: coordinates of image B
        """
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA[i] = KptA[m.queryIdx].pt
            ptsB[i] = KptB[m.trainIdx].pt
        return ptsA, ptsB

    @classmethod
    def find_homography(cls,kpsA, kpsB, matches, method=0):
        """
        calculate homography matrix (perspective transformation), should have at least 4 corresponding point
        H: homography matrix
        """
        print('find_homography =>')
        if method == 0:
            method = cv2.RANSAC
        else:
            method = cv2.RANSAC
            print('[{}] Only algorithm `{}` can be used so far, sorry ~ '.format(cls.__name__))

        if matches is None:
            print('matches is None')
        if kpsA is None:
            print('kpsA is None')
        if kpsB is None:
            print('kpsB is None')
        ptsA, ptsB = cls.Match2Keypoint(matches,kpsA,kpsB)
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=method)
        print('find_homography => Done')
        return H

    @classmethod
    def wrap(self, image, H, size ):
        """
        align image via transformation
        wraped: wraped image
        """
        # if( len(image.shape) > 2 ):
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        wraped = cv2.warpPerspective(image, H, size)
        return wraped

    def execute(self, data: Any) -> Any:
        print('ImageAlignment.execute => '+ str(data))
        Data.check_type(data.query_image,np.ndarray)
        Data.check_type(data.template_image,np.ndarray)
        Data.check_type(data.query_keypoints,list)
        Data.check_type(data.template_keypoints,list)
        Data.check_type(data.matches,list)

        H_matrix = ImageAlignment.find_homography(data.query_keypoints , data.template_keypoints, data.matches)
        aligned_img = self.wrap(data.query_image, H_matrix, LoadImage().get_size(data.template_image) )
        WriteImage().execute(aligned_img, save_path = './aligned.png')
        data.aligned_image = aligned_img
        return data


class WriteImage:
    def execute(self,image: np.ndarray,save_path: str = './') -> None:
        cv2.imwrite(save_path,image)


class LoadImage:
    def execute(self, data: Any) -> Any:
        print('LoadImage.execute =>'+ str(data))
        
        # key = list(img_path.keys())[0]
        # value = list(img_path.values())[0]
        # input_img = cv2.imread(value)
        # _data = Data()
        # _data._DATA[key]=value
        # _data._DATA['query_image']=input_img
        if not Data.check_exists(data.query_image):
            print('[{}] `query_image` is not existed, loading it now...'.format(self.__class__.__name__))
            input_img = cv2.imread(data.queryImg_path)
            data.query_image = input_img
        elif not Data.check_exists(data.template_image):
            print('[{}] `template_image` is not existed, loading it now...'.format(self.__class__.__name__))
            input_img = cv2.imread(data.templateImg_path)
            data.template_image = input_img
        else:
            print('[{}] `query_image` and `template_image` are already existed. '.format(self.__class__.__name__)) 

        return data

    def get_size(self,image):
        return image.shape[:2]


class PipelineBase:

    def __init__(self, 
                 processes: List[Any], 
                 *args, **kwargs) -> None:
        self._processes = processes
        # self.processes = []
        for k, v in kwargs.items():
            self[k] = v
        # print('self.processes=>'+str(self.processes) )

    def execute(self, in_out: Any):
       
        for _proc in self._processes:
            print(type(_proc))

            if inspect.isclass(_proc):
                proc = self._init_class(_proc)
            else:
                proc = _proc

            print('proc =>')
            print(proc)
            # self.processes.append(proc)
          
            in_out = proc.execute(in_out)
            print('in_out => '+str(type(in_out)))

        return in_out

    def _init_class(self, proc):
        print('_init_class')
        sig = inspect.signature(proc)
        print('signature=> '+str(sig))
        kwargs = {}
        for k in sig.parameters.keys():
            if k == 'self':
                continue

            v = getattr(self, k)
            kwargs[k] = v

        return proc(**kwargs)

@dataclass
class Data:

    queryImg_path: str = './'
    templateImg_path: str = './'
    query_image: np.ndarray = None
    template_image: np.ndarray = None
    query_keypoints: list = None
    query_descriptors: np.ndarray = None
    template_keypoints: list = None
    template_descriptors: np.ndarray = None
    matches: list = None
    aligned_image: np.ndarray = None

    @classmethod
    def check_exists(cls,field_name):
        if field_name is None:
            return False
        return True

    @classmethod
    def check_type(cls,field_name,field_type):
        print('check_type =>')
        if not cls.check_exists(field_name):
            print('[{}]: field_name is None =>'.format(cls.__name__)) 
            raise Exception('[{}]: The field is not existed ! '.format(cls.__name__))

        if isinstance(field_name,field_type):
            print('[{}]: Datatype {} is correct !'.format(cls.__name__,field_type))
        else:
            raise Exception('[{}]: Datatype {} is incorrect ! '.format(cls.__name__,field_type))






if __name__ == '__main__':
    """
    experiment pipeline for image registration

    """
    print('[START] pipeline testing ------------------------------')
    processes = [LoadImage, FeatureExtraction(maxFeatures=200,keepPercent=0.5),LoadImage,FeatureExtraction(maxFeatures=200,keepPercent=0.5),FeatureMatching(keepPercent=0.5),ImageAlignment]
    vision_pipeline = PipelineBase(processes)
    pipeline_data = Data()
    pipeline_data.queryImg_path = './image/table4.jpg'
    pipeline_data.templateImg_path = './image/template.jpg'
    pipeline_data = vision_pipeline.execute(pipeline_data)
    print('pipeline_data =>'+str(pipeline_data))
