from typing import List, Any, Union, Tuple
import inspect
import cv2
import numpy as np
from feature_extraction import FeatureExtraction,ImageAlignment, Display


class write_image:
    def execute(self,image: np.ndarray,save_path: str = './') -> None:
        cv2.imwrite(save_path,image)

class load_image:
    
    def execute(self, img_path: str) -> np.ndarray:
        print('LoadImage.execute =>'+img_path)
        ## TODO: 參數設定color or gray image
        input_img = cv2.imread(img_path)
        # input_img ='abc'
        return input_img

    def get_size(self,image):
        return image.shape[:2]


class feature_extraction:
    
    def __init__(self,
        maxFeatures: int = 200,
        keepPercent: float = 0.5) -> None:
        print('feature_extraction.__init__')
        # Parameters of feature extraction
        self.maxFeatures = maxFeatures
        self.keepPercent = keepPercent
        
    def execute(self, image: np.ndarray) -> Tuple[tuple,np.ndarray]:
        print('FeatureExtraction.execute ')
        (keypoints, descrips) = FeatureExtraction().get_keypoint(image,maxFeatures=self.maxFeatures)
        Display().show_keypoints(keypoints)
        return keypoints, descrips

class feature_matching:
    # def __init__(self,
    #     descripsQuery: tuple,
    #     descripsReference: np.ndarray,
    #     keepPercent: float = 0.5) -> None:
    #     if descripsQuery is None:
    #         raise Exception(' Can NOT find `descripsQuery` ')
        
    #     if descripsReference is None:
    #         raise Exception(' Can NOT find `descripsReference` ')
    #     self.descripsQuery = descripsQuery
    #     self.descripsReference = descripsReference
    #     self.keepPercent = keepPercent
    def __init__(self,keepPercent: float = 0.5) -> None:
        self.keepPercent = keepPercent

    def execute(self, data: Any) -> list:
        print('FeatureMatching.execute ')
        matches = ImageAlignment().match(data[0],data[1], keepPercent=self.keepPercent)
        return matches

class image_alignment:
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
   
    def execute(self, img_path: str) -> np.ndarray:
        print('ImageAlignment.execute ')
        print('is subclass ? '+str(issubclass(image_alignment, load_image)))
        input_img = load_image().execute(img_path)
        template_img = load_image().execute(self.template_path)
        H_matrix = ImageAlignment().find_homography(self.query_keypoints , self.reference_keypoints, self.matches)
        aligned_img = ImageAlignment().wrap(input_img, H_matrix, load_image().get_size(template_img) )
        write_image().execute(aligned_img, save_path = './aligned.png')
        return aligned_img


class PipelineBase:

    # _PREPROC_CLASS_MAP = {
    #     'LoadImage': LoadImage,
    #     'FeatureExtraction': FeatureExtraction,
    #     'FeatureMatching': FeatureMatching,
    #     'ImageAlignment': ImageAlignment,
    # }
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
            # TODO: 若類別則為物件做初始化，若不是維持原狀，最後
            if inspect.isclass(_proc):
                proc = self._init_class(_proc)
            else:
                proc = _proc

            print('proc =>')
            print(proc)
            # self.processes.append(proc)
          
            in_out = proc.execute(in_out)
            print('text=> '+str(type(in_out)))

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


if __name__ == '__main__':
    # processes = [LoadImage,FeatureExtraction,FeatureMatching,ImageAlignment]
    processes_LocalFeatures = [load_image,feature_extraction(maxFeatures=200,keepPercent=0.5)]
    vision_pipeline = PipelineBase(processes_LocalFeatures)
    # get features on query image
    img_path = "./image/table4.jpg"
    output = vision_pipeline.execute(img_path)
    query_keypoints = output[0]
    query_descriptors = output[1]
    print('get output from Pipeline =>')
    print('Total {} keypoints and {} descriptors'.format(len(query_keypoints),len(query_descriptors)))
    Display().show_keypoints(query_keypoints)
    # Display().show_descriptors(query_descriptors)
    # get features on  image
    img_path = "./image/template.jpg"
    output = vision_pipeline.execute(img_path)
    reference_keypoints = output[0]
    reference_descriptors = output[1]
    print('Total {} keypoints and {} descriptors'.format(len(reference_keypoints),len(reference_descriptors)))
    Display().show_keypoints(reference_keypoints)
    # Display().show_descriptors(reference_descriptors)
    # feature matching
    processes_alignment = [feature_matching(keepPercent=0.5)]
    vision_pipeline = PipelineBase(processes_alignment)
    matches = vision_pipeline.execute([query_descriptors,reference_descriptors])
    Display().show_matches(matches)

    # alignment
    img_path = "./image/table4.jpg"
    template_path = "./image/template.jpg"
    processes_alignment = [image_alignment(template_path,query_keypoints,reference_keypoints,matches)]
    vision_pipeline = PipelineBase(processes_alignment)
    result_img = vision_pipeline.execute(img_path)