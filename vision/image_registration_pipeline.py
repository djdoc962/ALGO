from typing import Dict, List, Any, Union, Tuple, get_type_hints
from dataclasses import dataclass, asdict, make_dataclass, is_dataclass
import inspect
import cv2
import numpy as np

class Display:
    def show_keypoints(self,keypoints):
        print('keypoints length: {} ==================================================================='.format(len(keypoints)))
        for i, keypoint in enumerate(keypoints):
            print('[KEYPOINTS]: {}, (x,y): ({}, {})'.format(i,keypoint.pt[0],keypoint.pt[1]))

    def show_descriptors(self,descriptors):
        print('descriptors length: {} with dimension: {} ==================================================================='.format(len(descriptors),len(descriptors[0])))
        for i, feature in  enumerate(descriptors):
            print('[DESCRIPTORS]: {}, feature: {}'.format(i,feature[:]))

    def show_matches(self,matches):
        """
        matches: DMatch object from OpenCV
        match.trainIdx: Index of the descriptor in train descriptors
        match.queryIdx: Index of the descriptor in query descriptors
        match.distance: Distance between descriptors. The lower, the better it is.
        DMatch.imgIdx: Index of the train image
        TODO: knnMatch, will be [[<DMatch>,<DMatch>],[<DMatch>,<DMatch>],...]
        """
        print('matches length: {} ==================================================================='.format(len(matches)))
        for i, match in enumerate(matches):
            print('[MATCHES]: {}, Idx: {}, {}'.format(i,match.queryIdx,match.trainIdx))


class FeatureMatching:
    def __init__(self,keepPercent: float = 0.5, method: int = 0) -> None:
        self.keepPercent = keepPercent
        self.method = method

    def execute(self, data: Any) -> Any:
        print('[{}] The processing of features matching is started ...'.format(self.__class__.__name__))
        if not Data.check_type(data.query_descriptors,np.ndarray):
            raise Exception('[{}] `query_descriptors` must be {} !'.format(self.__class__.__name__,Data.get_type('query_descriptors')))

        if not Data.check_type(data.template_descriptors,np.ndarray):
            raise Exception('[{}] `template_descriptors` must be {} !'.format(self.__class__.__name__,Data.get_type('template_descriptors')))

        
        matches = self.match(data.query_descriptors,data.template_descriptors, keepPercent=self.keepPercent,method=self.method)
        data.matches = matches
        print('[{}] The processing of features matching is finished.'.format(self.__class__.__name__))
        return data

    @classmethod
    def good_matches(cls,matches):
        # Apply ratio test as Lowe's paper
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return good
    
    @classmethod
    def match(cls, descripA, descripB,keepPercent=0.2, method=0):
        """
        MatcherType {
         FLANNBASED = 1,
         BRUTEFORCE = 2,
         BRUTEFORCE_L1 = 3,
         BRUTEFORCE_HAMMING = 4,
         BRUTEFORCE_HAMMINGLUT = 5,
         BRUTEFORCE_SL2 = 6
        }
        match the features between images
        matches: DMatch object from OpenCV
        """
        if method == 0:
            method = cv2.DescriptorMatcher_BRUTEFORCE_HAMMING
            print('[{}] Using BRUTEFORCE_HAMMING:{} '.format(cls.__name__,method))
            matcher = cv2.DescriptorMatcher_create(method)
            matches = matcher.match(descripA, descripB, None)
            matches = sorted(matches, key=lambda x:x.distance)
            # keep only the top matches
            keep = int(len(matches) * keepPercent)
            matches = matches[:keep]
        else:
            K = 2
            #TODO: or get k best matches via knnMatch, which will be [[<DMatch>,<DMatch>],[<DMatch>,<DMatch>],...]
            method = cv2.DescriptorMatcher_FLANNBASED
            print('[{}] Using FLANN:{} '.format(cls.__name__,method))
            matcher = cv2.DescriptorMatcher_create(method)
            # FLANN parameters
            FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(descripA, descripB,k=K)
      
        return matches


class ImageAlignment:

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
        print('[{}] The processing of homography calculation is started ...'.format(cls.__name__))
        if method == 0:
            method = cv2.RANSAC
        else:
            method = cv2.RANSAC
            print('[{}] Only algorithm `{}` can be used so far, sorry ~ '.format(cls.__name__))

        if matches is None:
            raise Exception('[{}] `matches` can not be found !'.format(cls.__name__))
            
        if kpsA is None:
            raise Exception('[{}] `kpsA` can not be found !'.format(cls.__name__))
        if kpsB is None:
            raise Exception('[{}] `kpsB` can not be found !'.format(cls.__name__))
        
        ptsA, ptsB = cls.Match2Keypoint(matches,kpsA,kpsB)
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=method)
        print('[{}] The processing of homography calculation is done.'.format(cls.__name__))
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
        print('[{}] Image alignment is started ...'.format(self.__class__.__name__)) 
        if not Data.check_type(data.query_image,np.ndarray):
            raise Exception('[{}] `query_image` must be {} !'.format(self.__class__.__name__,Data.get_type('query_image')))

        if not Data.check_type(data.template_image,np.ndarray):
           raise Exception('[{}] `template_image` must be {} !'.format(self.__class__.__name__,Data.get_type('template_image')))

        if not Data.check_type(data.query_keypoints,list):
            raise Exception('[{}] `query_keypoints` must be {} !'.format(self.__class__.__name__,Data.get_type('query_keypoints')))

        if not Data.check_type(data.template_keypoints,list):
            raise Exception('[{}] `template_keypoints` must be {} !'.format(self.__class__.__name__,Data.get_type('template_keypoints')))

        if not Data.check_type(data.matches,list):
            raise Exception('[{}] `matches` must be {} !'.format(self.__class__.__name__,Data.get_type('matches')))


        H_matrix = ImageAlignment.find_homography(data.query_keypoints , data.template_keypoints, data.matches)
        aligned_img = self.wrap(data.query_image, H_matrix, LoadImage().get_size(data.template_image) )
        print('[{}] Image alignment is finished.'.format(self.__class__.__name__)) 
        WriteImage().execute(aligned_img, save_path = './aligned.png')
        data.aligned_image = aligned_img
        
        return data


class WriteImage:
    def execute(self,image: np.ndarray,save_path: str = './result.png') -> None:
        print('[{}] Writing image to {} ...'.format(self.__class__.__name__,save_path))  
        cv2.imwrite(save_path,image)
        print('[{}] Image is saved to {}.'.format(self.__class__.__name__,save_path)) 


class FeatureExtraction:
    def __init__(self,
        maxFeatures: int = 200,
        method: int  = 0) -> None:
        self.maxFeatures = maxFeatures
        self.method = method
     
        
    def execute(self, data: Any) -> Any:
        if not Data.check_exists(data,'input_image'):
            raise Exception('[{}] `input_image` can not be found ! '.format(self.__class__.__name__)) 

        print('[{}] Feature extraction is started ...'.format(self.__class__.__name__))    
        (keypoints, descrips) = self.get_keypoint(data.input_image,maxFeatures=self.maxFeatures,method=self.method)
        data.input_keypoints = keypoints
        data.input_descriptors = descrips   

        if not Data.check_type(data.input_keypoints,list):
            raise Exception('[{}] `input_keypoints` must be {} !'.format(self.__class__.__name__,Data.get_type('input_keypoints')))

        if not Data.check_type(data.input_descriptors,np.ndarray):
            raise Exception('[{}] `input_descriptors` must be {} !'.format(self.__class__.__name__,Data.get_type('input_descriptors')))


        print('[{}] Feature extraction is finished.'.format(self.__class__.__name__))

        return data


    def get_keypoint(self, image, maxFeatures=500, method=0):
        """
        detect keypoints and extract (binary) local invariant features
        keypoints: FAST keypoint including coordination
        descrips: BRIEF descriptor(32 dimensions By default)
        """
        # convert both the input image to grayscale
        if( len(image.shape) > 2 ):
            print('[{}] It is a color image, will be converted to gray image.'.format(self.__class__.__name__))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        if method == 0:
            orb = cv2.ORB_create(maxFeatures)
            (keypoints, descrips) = orb.detectAndCompute(image, None)
        else:    
            orb = cv2.ORB_create(maxFeatures)
            (keypoints, descrips) = orb.detectAndCompute(image, None)

        return keypoints, descrips


class LoadImage:
    def execute(self, data: Any) -> Any:
        if not Data.check_exists(data,'img_path'):
            raise Exception('[{}] `img_path` can not be found !'.format(self.__class__.__name__))
       
        if not Data.check_type(data.img_path,str):
            raise Exception('[{}] `img_path` must be {} !'.format(self.__class__.__name__,Data.get_type('img_path')))
        
        print('[{}] Loading image from {} ...'.format(self.__class__.__name__,data.img_path))  
        input_img = cv2.imread(data.img_path)
        data.input_image = input_img   
        print('[{}] The processing of image loading is finished.'.format(self.__class__.__name__)) 
        return data

    def get_size(self,image):
        return image.shape[:2]


class PipelineBase:

    def __init__(self, 
                 processes: List[Any], 
                 *args, **kwargs) -> None:
        self._processes = processes
        for k, v in kwargs.items():
            self[k] = v
 

    def execute(self, in_out: Any):
        for _proc in self._processes:
            if inspect.isclass(_proc):
                proc = self._init_class(_proc)
            else:
                proc = _proc
          
            in_out = proc.execute(in_out)

        return in_out

    def _init_class(self, proc):
        sig = inspect.signature(proc)
        kwargs = {}
        for k in sig.parameters.keys():
            if k == 'self':
                continue

            v = getattr(self, k)
            kwargs[k] = v

        return proc(**kwargs)


@dataclass
class Data:

    """
    specified data list for data storing in image processes
    """    
    img_path: str = None
    # input_image: np.ndarray = np.array([])
    input_image: np.ndarray = None
    input_keypoints: list = None
    input_descriptors: np.ndarray = None  
    queryImg_path: str = None
    templateImg_path: str = None
    query_image: np.ndarray = None
    template_image: np.ndarray = None
    query_keypoints: list = None
    query_descriptors: np.ndarray = None
    template_keypoints: list = None
    template_descriptors: np.ndarray = None
    matches: list = None
    aligned_image: np.ndarray = None
    handles: object = None
    

    def __init__(self,clsname: str, memberList: list) -> None:
        if clsname is None:
            raise Exception('[{}] `clsname` can not be None !'.format(self.__class__.__name__))
        if memberList is None:
            raise Exception('[{}] `memberList` can not be None !'.format(self.__class__.__name__))
        
        if not Data.check_type(clsname, str):
            raise Exception('[{}] `clsname` must be {} ! '.format(self.__class__.__name__,type(str)))

        if not Data.check_type(memberList, list):
            raise Exception('[{}] `memberList` must be {} ! '.format(self.__class__.__name__,type(list))) 
        
        memberType = []
        memberValue = []
        #datatype and values collection
        for item in memberList:
            #check if the member is valid 
            if not hasattr(Data,item[0]):
                raise Exception('[{}]: `{}` is not valid ! '.format(self.__class__.__name__,item[0]))
            
            dataType = type(None)
            if item[1] is not None:
                #check if the member datatype is valid 
                if not (Data.get_type(item[0]) == type(item[1])):
                    raise Exception('[{}] `{}` must be {} !'.format(self.__class__.__name__,item[0],Data.get_type(item[0])))

                if type(item[1]).__name__ == 'str':
                    dataType = str
                elif  type(item[1]).__name__ == 'list':
                    dataType = list 
                elif type(item[1]).__name__ == 'ndarray':
                    dataType = np.ndarray
                else:
                    raise Exception('[{}]: Datatype `{}` is not valid ! '.format(self.__class__.__name__,type(item[1]).__name__))
                    
            memberType.append((item[0],dataType))
            memberValue.append((item[0],item[1]))

        # create new dataclass with member datatype and values
        self.handles = make_dataclass(clsname,memberType)
        for value in memberValue:
            # print('(fieldName: {}, value: {} )'.format(value[0],type(value[1])))
            setattr(self.handles,value[0],value[1])
        
 
    def get_data(self):
        return self.handles
  

    @classmethod
    def check_exists(cls,data,field_name):
        if not hasattr(data,field_name):
            return False
        return True

    @classmethod
    def get_type(cls,field_name):
        attrs = get_type_hints(Data)
        return attrs[field_name] 

    @classmethod
    def check_type(cls,field_value,field_type):
        if not isinstance(field_value,field_type):
            return False
        return True

               
class LocalFeaturesPairs:
    def __init__(self, maxFeatures: int = 200, method:int = 1) -> None:
        self.maxFeatures = maxFeatures
        self.method = method
    
    def execute(self, data: Any) -> Any:
        print('[{}] Extracting the pairs of features from query image and template image ...'.format(self.__class__.__name__))  
        if not Data.check_exists(data,'queryImg_path'):
            raise Exception('[{}] `queryImg_path` can not be found !'.format(self.__class__.__name__))

        if not Data.check_exists(data,'templateImg_path'):
            raise Exception('[{}] `templateImg_path` can not be found !'.format(self.__class__.__name__))

        if not Data.check_type(data.queryImg_path,str):
            raise Exception('[{}] `queryImg_path` must be {} !'.format(self.__class__.__name__,Data.get_type('queryImg_path')))

        if not Data.check_type(data.templateImg_path,str):
            raise Exception('[{}] `templateImg_path` must be {} !'.format(self.__class__.__name__,Data.get_type('templateImg_path')))

       
        _data = Data('_data',[('img_path',data.queryImg_path)]).get_data()
        processes = [LoadImage,FeatureExtraction(maxFeatures=self.maxFeatures,method=self.method)]
        vision_pipeline = PipelineBase(processes)
        _data = vision_pipeline.execute(_data)
        # print('query image data=>'+str(_data))
        data.query_image = _data.input_image
        data.query_keypoints = _data.input_keypoints
        data.query_descriptors = _data.input_descriptors

        _data.img_path = data.templateImg_path
        vision_pipeline = PipelineBase(processes)
        _data = vision_pipeline.execute(_data)
        # print('template image data=>'+str(_data))
        data.template_image = _data.input_image
        data.template_keypoints = _data.input_keypoints
        data.template_descriptors = _data.input_descriptors
        print('[{}] The processing of pairs of features extraction is done.'.format(self.__class__.__name__))  
        return data




if __name__ == '__main__':
    """
    experiment pipeline for image registration

    """
    print('[START] Experiment Pipeline testing ------------------------------')
    print('- Specified data list: ')
    print(get_type_hints(Data))
  
    #EX1. Just load an image
    processes = [LoadImage]
    _LoadImageData = Data('LoadImageData',[('img_path','./image/box.png'),('input_image',None)]).get_data()
    # print('Is {} a dataclass ? {}'.format(_LoadImageData.__name__,str(is_dataclass(_LoadImageData))))
    vision_pipeline = PipelineBase(processes)
    _LoadImageData = vision_pipeline.execute(_LoadImageData)
    # print('_LoadImageData.input_image => ')
    # print(str(_LoadImageData.input_image))

    #EX2. Just extract keypoints and descriptors from the previously loaded image
    _FeatureExtractionData = Data('FeatureExtractionData',[('input_image',_LoadImageData.input_image),('input_keypoints',None),('input_descriptors',None)]).get_data()
    processes = [FeatureExtraction(maxFeatures=200,method=0)]
    vision_pipeline = PipelineBase(processes)
    _FeatureExtractionData = vision_pipeline.execute(_FeatureExtractionData)
    # print('_FeatureExtractionData.input_keypoints => ')
    # Display().show_keypoints(_FeatureExtractionData.input_keypoints)

    # EX3. Just extract keypoints and descriptors from the previously loaded image 
    _LoadImage_FeatureExtraction = Data('LoadImage_FeatureExtraction',[('img_path','./image/box.png'),('input_image',None),('input_keypoints',None),('input_descriptors',None)]).get_data()
    print('_LoadImage_FeatureExtraction => '+str(_LoadImage_FeatureExtraction)) 
    processes = [LoadImage,FeatureExtraction(maxFeatures=200,method=0)]
    vision_pipeline = PipelineBase(processes)
    _LoadImage_FeatureExtraction = vision_pipeline.execute(_LoadImage_FeatureExtraction)
    # print('_LoadImage_FeatureExtraction.input_keypoints => ')
    # Display().show_keypoints(_LoadImage_FeatureExtraction.input_keypoints)
    
    #EX4. A pairs of image features generated by LocalFeaturesPairs, then do feature matching and alignment 
    da = 'dfd' 
    PairData = Data('PairData',[('queryImg_path','./image/insurance_query.jpg'),('templateImg_path','./image/insurance_template.jpg')]).get_data()
    vision_pipeline = PipelineBase([LocalFeaturesPairs(maxFeatures=200,method=0),FeatureMatching(keepPercent=0.5,method=0),ImageAlignment])
    PairData = vision_pipeline.execute(PairData)
    # print('PairData => ')
    # Display().show_keypoints(PairData.query_keypoints)
    # Display().show_descriptors(PairData.query_descriptors)
    # Display().show_keypoints(PairData.template_keypoints)
    # Display().show_descriptors(PairData.template_descriptors)
    # Display().show_matches(PairData.matches)
  

  



    
    