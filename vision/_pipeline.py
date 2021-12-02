from typing import List, Any, Union
import inspect


class LoadImage:
    def execute(self):
        print('LoadImage.execute ')

class FeatureExtraction:
    def execute(self):
        print('FeatureExtraction.execute ')

class FeatureMatching:
    def execute(self):
        print('FeatureMatching.execute ')

class ImageAlignment:
    def execute(self):
        print('ImageAlignment.execute ')


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
        self.processes = []
        for k, v in kwargs.items():
            self[k] = v
        print('self.processes=>'+str(self.processes) )

    def execute(self):
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
            self.processes.append(proc)
          
            proc.execute()

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
    processes = [LoadImage,FeatureExtraction,FeatureMatching,ImageAlignment]
    data_pipeline = PipelineBase(processes)
    data_pipeline.execute()
