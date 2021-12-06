from typing import List, Any, Union, Tuple
import inspect
import cv2
import numpy as np
from feature_extraction import FeatureExtraction,ImageAlignment, Display


class load_image:
    
    def execute(self, img_path: str) -> np.ndarray:
        print('LoadImage.execute ')
        ## TODO: 參數設定color or gray image
        input_img = cv2.imread(img_path)
        # input_img ='abc'
        return input_img

class feature_extraction:
    
    def __init__(self,
        maxFeatures: int = 200,
        keepPercent: float = 0.5) -> None:
        print('feature_extraction.__init__')
        self.process = FeatureExtraction()
        # Parameters of feature extraction
        self.maxFeatures = maxFeatures
        self.keepPercent = keepPercent
        
        
    def execute(self, image: object) -> Tuple[tuple,np.ndarray]:
        print('FeatureExtraction.execute ')
        (keypoints, descrips) = self.process.get_keypoint(image,maxFeatures=self.maxFeatures)
        Display().show_keypoints(keypoints)
        return keypoints, descrips

class feature_matching:
    def execute(self, text: str) -> str:
        print('FeatureMatching.execute ')
        return text+'_FeatureMatching'

class image_alignment:
    def execute(self, text: str) -> str:
        print('ImageAlignment.execute ')
        return text+'_ImageAlignment'


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
            # if isinstance(_proc, LoadImage):
            #     proc = self._init_class(_proc)
            # else:
            #     proc = _proc

            if inspect.isclass(_proc):
                proc = self._init_class(_proc)
            else:
                proc = _proc

            print('proc =>')
            print(proc)
            # self.processes.append(proc)
          
            in_out = proc.execute(in_out)
            print('text=> '+str(type(in_out)))

        return self

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
    processes = [load_image,feature_extraction(maxFeatures=200,keepPercent=0.5)]
    vision_pipeline = PipelineBase(processes)
    img_path = "./image/table4.jpg"
    vision_pipeline.execute(img_path)
