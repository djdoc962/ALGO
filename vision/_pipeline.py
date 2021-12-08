from typing import Dict, List, Any, Union, Tuple
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
        
    def execute(self, data: Any) -> Tuple[tuple,np.ndarray]:
        print('FeatureExtraction.execute ')
        # image = data['query_image']
        # print(image)
        (keypoints, descrips) = FeatureExtraction().get_keypoint(data['query_image'],maxFeatures=self.maxFeatures)
        # print('show_keypoints =>')
        # Display().show_keypoints(keypoints)
        data['query_keypoints'] = keypoints
        data['query_descriptors'] = descrips
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

    def execute(self, data: Any) -> list:
        print('FeatureMatching.execute ')
        matches = ImageAlignment.match(data[0],data[1], keepPercent=self.keepPercent)
        return matches
 

class ImageAlignment:
    def __init__(self,template_path: str,query_keypoints: tuple, reference_keypoints: tuple, matches: list) -> None:
        print('image_alignment.__init__')
        if template_path is None:
            raise Exception(' Can NOT find `template_path` ')
        if query_keypoints is None:
            raise Exception(' Can NOT find `query_keypoints` ')
        if reference_keypoints is None:
            raise Exception(' Can NOT find `reference_keypoints` ')
        if matches is None:
            raise Exception(' Can NOT find `matches` ')
        self.template_path = template_path
        self.query_keypoints = query_keypoints
        self.reference_keypoints = reference_keypoints
        self.matches = matches
   
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

        if matches is None:
            print('matches is None')
        if kpsA is None:
            print('kpsA is None')
        if kpsB is None:
            print('kpsB is None')
        ptsA, ptsB = cls.Match2Keypoint(matches,kpsA,kpsB)
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
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

    def execute(self, img_path: str) -> np.ndarray:
        print('ImageAlignment.execute ')
    
        input_img = LoadImage().execute(img_path)
        template_img = LoadImage().execute(self.template_path)
        a = ImageAlignment
        H_matrix = a.find_homography(self.query_keypoints , self.reference_keypoints, self.matches)
        aligned_img = self.wrap(input_img, H_matrix, LoadImage().get_size(template_img) )
        WriteImage().execute(aligned_img, save_path = './aligned.png')
        return aligned_img


class WriteImage:
    def execute(self,image: np.ndarray,save_path: str = './') -> None:
        cv2.imwrite(save_path,image)


class LoadImage:
    def execute(self, img_path: Any) -> np.ndarray:
        print('LoadImage.execute =>'+ str(img_path))
        # input_img = cv2.imread(img_path)
        key = list(img_path.keys())[0]
        value = list(img_path.values())[0]
        input_img = cv2.imread(value)
        _data = Data()
        _data._DATA[key]=value
        _data._DATA['query_image']=input_img
        return _data._DATA

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
        out_data = []
        for _proc in self._processes:
            print(type(_proc))
            # TODO: 若類別則為物件做初始化，若不是維持原狀，最後
            
            if inspect.isclass(_proc):
                proc = self._init_class(_proc)
            else:
                proc = _proc

            print('proc =>')
            print(proc)
            # self.processes.append(proc)
          
            in_out = proc.execute(in_out)
            out_data.append(in_out)
            print('text=> '+str(type(in_out)))

        return out_data

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


class Data:
    def __init__(self) -> None:
        self._DATA_MAP = {
        'queryImg_path': str,
        'templateImg_path': str,
        'query_image': np.ndarray,
        'template_image': np.ndarray,
        'query_keypoints': tuple,
        'query_descriptors': tuple,
        'template_Keypoints': tuple,
        'template_descriptors': tuple,
        'matches': list,
        }
        
        self._DATA = {}

if __name__ == '__main__':
    # processes = [LoadImage,FeatureExtraction,FeatureMatching,ImageAlignment]
    processes_LocalFeatures = [LoadImage,FeatureExtraction(maxFeatures=200,keepPercent=0.5)]
    vision_pipeline = PipelineBase(processes_LocalFeatures)
    # get features on query image
    img_path = "./image/table4.jpg"
    #TODO: 每個process的input/output用Dictionary來傳遞
    output = vision_pipeline.execute( {'queryImg_path':img_path})
    output = output[0]
    print('output type : '+str(type(output)))
    for key in output.keys():
        # print(key,'->',output[0][key])
        print('Key->'+ key)
       
    # query_keypoints = output[0]
    # query_descriptors = output[1]
    query_keypoints = output['query_keypoints']
    query_descriptors = output['query_descriptors']
    print('get output from Pipeline =>')
    print('Total {} keypoints and {} descriptors'.format(len(query_keypoints),len(query_descriptors)))
    Display().show_keypoints(query_keypoints)
    # Display().show_descriptors(query_descriptors)
    # get features on image
    img_path = "./image/template.jpg"
    output = vision_pipeline.execute({'templateImg_path':img_path})
    output = output[0]
    print('output type : '+str(type(output)))
    for key in output.keys():
        # print(key,'->',output[0][key])
        print('Key->'+ key)
    # reference_keypoints = output[0]
    # reference_descriptors = output[1]
    reference_keypoints = output['template_keypoints']
    reference_descriptors = output['template_descriptors']

    print('Total {} keypoints and {} descriptors'.format(len(reference_keypoints),len(reference_descriptors)))
    Display().show_keypoints(reference_keypoints)
    # Display().show_descriptors(reference_descriptors)
    # feature matching
    processes_alignment = [FeatureMatching(keepPercent=0.5)]
    vision_pipeline = PipelineBase(processes_alignment)
    matches = vision_pipeline.execute([query_descriptors,reference_descriptors])
    Display().show_matches(matches)

    # alignment
    img_path = "./image/table4.jpg"
    template_path = "./image/template.jpg"
    processes_alignment = [ImageAlignment(template_path,query_keypoints,reference_keypoints,matches)]
    vision_pipeline = PipelineBase(processes_alignment)
    result_img = vision_pipeline.execute(img_path)