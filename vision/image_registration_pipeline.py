from typing import Dict, List, Any, Union, Tuple, get_type_hints
from dataclasses import dataclass, asdict, make_dataclass, is_dataclass
import inspect,os
import cv2
import numpy as np
from skimage.draw import line


class Display:

    def draw_circles(self, image, coordinates, color=(0,255,0), size=3, save_path: str = './draw_circles.png'):
        for xy in coordinates:
            cv2.circle(image,(int(xy[0][0]),int(xy[0][1])),size,color)
        WriteImage().execute(image,save_path)
        return image

    def draw_matches(self,query_image,kpsA,template_image, kpsB, matches, mode: int = 0, matchesMask:list = None, save_path: str = './draw_matches.png'):

        if mode == 0:
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2)

            matchedVis = cv2.drawMatches(query_image, kpsA, template_image, kpsB,
                matches, None,**draw_params)
        else:
            if not isinstance(matches[0],list):
                matches = [ [match] for match in matches]
            print('draw_matches.matches => '+str(matches))
            if matchesMask is not None:
                matchesMask = [ [item] for item in matchesMask]
                # print('matchesMask => '+str(matchesMask))
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2)


            matchedVis = cv2.drawMatchesKnn(query_image, kpsA, template_image, kpsB,
                matches, None,**draw_params)

        scale_percent =  150 # percent of original size
        width = int(matchedVis.shape[1] * scale_percent / 100)
        height = int(matchedVis.shape[0] * scale_percent / 100)
        dim = (width, height)
        # dim = self.resize_image(matchedVis,width=1000)

        matchedVis = cv2.resize(matchedVis, dim)
        WriteImage().execute(matchedVis,save_path)
        # cv2.imshow("Matched Keypoints..", matchedVis)
        # cv2.waitKey(0)


    def draw_keypoints(self, image, keypoints, save_path: str = './draw_keypoints.png'):
        img =cv2.drawKeypoints(image,keypoints,None, flags=0)
        WriteImage().execute(img,save_path)


    def show_keypoints(self,keypoints):
        print('keypoints length: {} ==================================================================='.format(len(keypoints)))
        for i, keypoint in enumerate(keypoints):
            print('[KEYPOINTS]: {}, (x,y): ({}, {})'.format(i,keypoint.pt[0],keypoint.pt[1]))

    def show_descriptors(self,descriptors):
        print('descriptors length: {} with dimension: {} ==================================================================='.format(len(descriptors),len(descriptors[0])))
        for i, feature in  enumerate(descriptors):
            print('[DESCRIPTORS]: {}, feature: {}'.format(i,feature[:]))

    def show_matches(self,matches,mode=0):
        """
        matches: DMatch object from OpenCV
        match.trainIdx: Index of the descriptor in train descriptors
        match.queryIdx: Index of the descriptor in query descriptors
        match.distance: Distance between descriptors. The lower, the better it is.
        DMatch.imgIdx: Index of the train image
        TODO: knnMatch, will be [[<DMatch>,<DMatch>],[<DMatch>,<DMatch>],...]
        """
        print('matches length: {} ==================================================================='.format(len(matches)))
        # print(str(matches))
        # if mode == 0:
        #     for i, match in enumerate(matches):
        #         print('[MATCHES]: {}, Idx of (Q,R), distance: ({}, {}) {}'.format(i,match.queryIdx,match.trainIdx,match.distance))
        # else:
        #     for i, match in enumerate(matches):
        #         for item in match:
        #             print('[MATCHES]: {}, Idx of (Q,R), distance: ({}, {}) {}'.format(i,item.queryIdx,item.trainIdx,item.distance))
        for i, match in enumerate(matches):
            if isinstance(match,list):
                match = match[0]
            print('[MATCHES]: {}, Idx of (Q,R), distance: ({}, {}) {}'.format(i,match.queryIdx,match.trainIdx,match.distance))


class Evaluation:
    def __init__(self,save_path: str ='./') -> None:
        self.save_path = save_path

    def precision(self) -> float:
        """
        Precision = correct matches/putative matches。
        """
        print('[{}] Calculating the precision ...'.format(self.__class__.__name__))
        precision = self.correct_matches_number/self.putative_matches_number
        print('[{}] Precision is done.'.format(self.__class__.__name__))
        return precision

    def recall(self) -> float:
        """
        Recall = correct matches/all matches
        """
        print('[{}] Calculating the recall ...'.format(self.__class__.__name__))
        recall = self.correct_matches_number/self.matches_number
        print('[{}] Recall is done.'.format(self.__class__.__name__))
        return recall

    def repeatability(self) -> float:
        """
        Repeatability = correct matches/minimum number of query keypoints and template keypoints
        """
        print('[{}] Calculating the repeatability ...'.format(self.__class__.__name__))
        repeatability = self.correct_matches_number/min(self.template_kps_number,self.query_kps_number)
        print('[{}] Repeatability is done.'.format(self.__class__.__name__))
        return repeatability

    @classmethod
    def keypoints2List(cls,kps) -> np.ndarray:
        """
        To convert KeyPoints to numpy array for perspetive transform
        """
        kps_list = []
        for i, keypoint in enumerate(kps):
            kps_list.append([keypoint.pt[0],keypoint.pt[1]])

        # print('keypoints2List get {} kps => {}'.format(len(kps_list),str(kps_list)))
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        pts = np.float32(kps_list).reshape(-1,1,2)
        return pts

    @classmethod
    def project_coordinates(cls, kps, M) -> list:
        """
        To project the keypoints coordinates to the template image using perspective transform
        kps: A numpy array of coordinates on image
        M: homography matrix
        dst_pts: projected coordinates
        """
        print('[{}] The processing of project_coordinates is started ...'.format(cls.__name__))

        if not Data.check_type(kps,np.ndarray):
            raise Exception('[{}] `kps` must be `numpy.ndarray` ! '.format(cls.__name__))

        dst_pts = cv2.perspectiveTransform(kps,M)
        print('[{}] The processing of project_coordinates is done.'.format(cls.__name__))
        return dst_pts

    @classmethod
    def coordinates2Keypoints(cls,pts) -> list:
        """
        To convert coordinates to KeyPoint object
        """
        keypoints = [cv2.KeyPoint(x[0][0], x[0][1],1) for x in pts]
        # Display().show_keypoints(keypoints)
        return keypoints

    @classmethod
    def bresenham(cls,x1,y1,x2,y2) -> list:
        coordinates = []
        m_new = 2 * (y2 - y1)
        slope_error_new = m_new - (x2 - x1)

        y=y1

        for x in range(x1,x2+1):
            coordinates.append([x,y])
            # print("(",x ,",",y ,")\n")

            # Slope error reached limit, time to
            # increment y and update slope error.
            if (slope_error_new >= 0):
                y=y+1
                slope_error_new =slope_error_new - 2 * (x2 - x1)

            # Add slope to increment angle formed
            slope_error_new =slope_error_new + m_new

        return coordinates

    @classmethod
    def get_correspondance(cls,ptsA,ptsB,err) -> list:
        """
        To get keypoint correspondances between template image and aligned query image based on points distance in pixels
        ptsA: numpy array of projected keypoints coordinates from query image
        ptsB: numpy array of keypoints coordinates from template image
        """
        print('[{}] The processing of correspondance is started ...'.format(cls.__name__))
        # Euclidean distance
        # dx = x1 - x2;
        # dy = y1 - y2;
        # dist = sqrt(dx * dx + dy * dy);
        correspondance = []

        for keypointA in ptsA:
            for keypointB in ptsB:
                rr, cc = line(int(keypointA[0][0]),int(keypointA[0][1]),int(keypointB[0][0]),int(keypointB[0][1]))
                # print('Distance is {} pixels'.format(len(list(zip(rr,cc)))))
                if len(list(zip(rr,cc))) < err:
                    correspondance.append([keypointB,keypointA])

        cls.correct_matches_number = len(correspondance)
        print('[{}] The processing of correspondance is done.'.format(cls.__name__))
        return correspondance

    def execute(self, data: Any) -> Any:
        ERROR = 1.5
        print('[{}] The processing of evaluation is started ...'.format(self.__class__.__name__))

        if not Data.check_type(data.query_keypoints,list):
            raise TypeError('[{}] `query_keypoints` must be {} !'.format(self.__class__.__name__,Data.get_type('query_keypoints')))

        if not Data.check_type(data.template_keypoints,list):
            raise TypeError('[{}] `template_keypoints` must be {} !'.format(self.__class__.__name__,Data.get_type('template_keypoints')))

        if not Data.check_exists(data,'matches'):
            raise ValueError('[{}] `matches` can not be found ! '.format(self.__class__.__name__))

        if not Data.check_type(data.matches,list):
            raise TypeError('[{}] `matches` must be {} !'.format(self.__class__.__name__,Data.get_type('matches')))

        if not Data.check_exists(data,'putative_matches'):
            raise ValueError('[{}] `putative_matches` can not be found ! '.format(self.__class__.__name__))

        if not Data.check_type(data.putative_matches,list):
            raise TypeError('[{}] `putative_matches` must be {} !'.format(self.__class__.__name__,Data.get_type('putative_matches')))

        if not Data.check_exists(data,'homography'):
            raise ValueError('[{}] `homography` can not be found ! '.format(self.__class__.__name__))

        if not Data.check_type(data.homography,np.ndarray):
            raise TypeError('[{}] `homography` must be {} !'.format(self.__class__.__name__,Data.get_type('homography')))

        try:
            ## If the matches have never be filtered, which can be seen as putative matches
            self.putative_matches_number = len(data.putative_matches)
            self.matches_number = len(data.matches)
            self.template_kps_number = len(data.template_keypoints)
            self.query_kps_number = len(data.query_keypoints)
            # extract query image keypoints from matches
            ptsA, ptsB = ImageAlignment.match2Keypoint(data.putative_matches,data.query_keypoints,data.template_keypoints)
            projPutativePts = self.project_coordinates(ptsA.reshape(-1,1,2),data.homography)

            # convert KeyPoints object to numpy array
            pts = self.keypoints2List(data.query_keypoints)
            projQueryPts = self.project_coordinates(pts,data.homography)

            pts = self.keypoints2List(data.template_keypoints)
            projTemplatePts = self.project_coordinates(pts,data.homography)

            templatePts = self.keypoints2List(data.template_keypoints)
            correct_matches = self.get_correspondance(projPutativePts,templatePts,ERROR)

            precision = self.precision()
            recall = self.recall()
            repeatability = self.repeatability()

            with open(os.path.join(self.save_path,'metrics.txt'), 'w') as file:
                line = 'Query image has total {} keypoints \n'.format(len(projQueryPts))
                line+= 'Template image has total {} keypoints \n'.format(len(templatePts))
                line+= 'Distance must be less than {} pixels \n'.format(ERROR)
                line+= 'Number of all matches is {}\n'.format(len(data.matches))
                line+= 'Number of putative matches is {}\n'.format(self.putative_matches_number)
                line+= 'Number of correct matches is {}\n'.format(len(correct_matches))
                line+= 'Precision: {:.2%}\n'.format(precision)
                line+= 'Recall: {:.2%}\n'.format(recall)
                line+= 'Repeatability: {:.2%}\n'.format(repeatability)
                file.write(line)

            print('- query image has {} keypoints'.format(len(projQueryPts)))
            print('- template image has {} keypoints '.format(len(templatePts)))
            print('- distance must less than {} pixels '.format(ERROR))
            print('- number of all matches is {}'.format(len(data.matches)))
            print('- number of putative matches is {}'.format(self.putative_matches_number))
            print('- number of correct matches is {}'.format(len(correct_matches)))
            print("- Precision: {:.2%}".format(precision))
            print("- Recall: {:.2%}".format(recall))
            print("- Repeatability: {:.2%}".format(repeatability))
        except Exception as e:
            raise Exception('[{}] Evaluation failed: {}'.format(self.__class__.__name__,str(e)))

        ## Drawing aligned results
        img = data.template_image.copy()
        img = Display().draw_circles(img,templatePts,color=(0,255,0),size=3,save_path =os.path.join(self.save_path,'origin_template.png'))
        img = Display().draw_circles(img,projTemplatePts,color=(0,0,255),size=3,save_path=os.path.join(self.save_path,'project_template.png'))
        img = Display().draw_circles(img,projQueryPts,color=(255,0,0),size=3,save_path=os.path.join(self.save_path,'project_query_template.png'))

        img = data.template_image.copy()
        correct_template = [ pair[0] for pair in correct_matches]
        correct_query = [pair[1] for pair in correct_matches]

        img = Display().draw_circles(img,correct_template,color=(255,0,0),size=2,save_path=os.path.join(self.save_path,'correct_matches.png'))
        img = Display().draw_circles(img,correct_query,color=(0,0,255),size=4,save_path=os.path.join(self.save_path,'correct_matches.png'))

        print('[{}] The processing of evaluation is done.'.format(self.__class__.__name__))


class FeatureMatching:
    def __init__(self,keepPercent: float = 0.5,filter: bool = True, method: int = 0, good_threshold: float = 0.75) -> None:
        self.keepPercent = keepPercent
        self.method = method
        self.filter = filter
        self.good_threshold = good_threshold

    def execute(self, data: Any) -> Any:
        print('[{}] The processing of features matching is started ...'.format(self.__class__.__name__))
        if not Data.check_type(data.query_descriptors,np.ndarray):
            raise Exception('[{}] `query_descriptors` must be {} !'.format(self.__class__.__name__,Data.get_type('query_descriptors')))

        if not Data.check_type(data.template_descriptors,np.ndarray):
            raise Exception('[{}] `template_descriptors` must be {} !'.format(self.__class__.__name__,Data.get_type('template_descriptors')))


        matches = self.match(data.query_descriptors,data.template_descriptors, keepPercent=self.keepPercent,method=self.method)
        # only get the closest element for KNN methods
        if self.method:
            new_matches = []
            for matche in matches:
                if len(matche)>0:
                    new_matches.append(matche[0])
            data.matches = new_matches
            ## get better matches by checking if ratio of distances is less the threshold
            if self.filter:
               matches = self.good_matches(matches, self.good_threshold)
               data.putative_matches = matches
            else:
               data.putative_matches = new_matches
        else:
            data.matches = matches
            if self.filter:
                # keep only the top matches
                keep = int(len(matches) * self.keepPercent)
                matches = matches[:keep]
                # data.putative_matches = matches
            data.putative_matches = matches

        print('[{}] The processing of features matching is finished.'.format(self.__class__.__name__))
        return data

    @classmethod
    def good_matches(cls, matches, good_threshold: float = 0.75):
        """
        To compute putative correspondences with distance ratio
        matches: DMatch object from OpenCV
        """
        print('[{}] To filter the matches using ratio of distance proposed by David Lowe ...'.format(cls.__name__))
        # Apply ratio test as Lowe's paper
        good = []
        for i, pair in enumerate(matches):
            if len(pair) < 2:
                print('[{}] Warning: The number of elements in matches < 2 .. '.format(cls.__name__))
                continue # you don't have the second element to compare against
            m,n = pair
            if m.distance < good_threshold*n.distance:
                good.append(m)

        print('[{}] Distance ratio test is done.'.format(cls.__name__))
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

        else:
            K = 2
            #TODO: or get k best matches via knnMatch, which will be [[<DMatch>,<DMatch>],[<DMatch>,<DMatch>],...]
            method = cv2.DescriptorMatcher_FLANNBASED
            print('[{}] Using FLANN:{} '.format(cls.__name__,method))
            # matcher = cv2.DescriptorMatcher_create(method)
            # FLANN parameters
            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(descripA, descripB,k=K)
            # keep only the top matches
            # keep = int(len(matches) * keepPercent)
            # matches = matches[:keep]

        # print(str(matches))
        return matches


class ImageAlignment:

    @classmethod
    def match2Keypoint(cls,matches,KptA,KptB):
        """
        get matching keypoints for each of the images
        KptA: list of KeyPoints of query image
        KptB: list of KeyPoints of template image
        ptsA: list of coordinates of query image
        ptsB: list of coordinates of template image
        """
        # print('matches => ')
        # print(str(len(matches)))
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
        To calculate homography matrix (perspective transformation), should have at least 4 corresponding point
        kpsA: a list of query image Keypoints object
        kpsB: a list of template image Keypoints object
        matches: a list of DMatch object
        method: RANSAC - RANSAC-based robust method
                LMEDS - Least-Median robust method
                RHO - PROSAC-based robust method
        H: homography matrix
        mask: a list of inliers/outliers mask
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

        ptsA, ptsB = cls.match2Keypoint(matches,kpsA,kpsB)
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=method)
        print('[{}] The processing of homography calculation is done.'.format(cls.__name__))
        return H, mask

    @classmethod
    def wrap(self, image, H, size ):
        """
        align image via transformation
        wraped: wraped image
        """
        wraped = cv2.warpPerspective(image, H, (size[1],size[0]))
        return wraped

    def execute(self, data: Any) -> Any:
        MIN_MATCH_COUNT = 4
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

        if len(data.putative_matches) >= MIN_MATCH_COUNT:
            print('ImageAlignment putative_matches =>'+str(data.putative_matches))
            H_matrix, mask = ImageAlignment.find_homography(data.query_keypoints , data.template_keypoints, data.putative_matches)
            matchesMask = mask.ravel().tolist()
            aligned_img = self.wrap(data.query_image, H_matrix, LoadImage().get_size(data.template_image) )
            print('[{}] Image alignment is finished.'.format(self.__class__.__name__))
            print('aligned image size: '+str(aligned_img.shape))
            data.homography = H_matrix
            data.aligned_image = aligned_img
            data.matchesMask = matchesMask
        else:
            raise Exception('[{}] `Minimum {} corresponding points` are required, only have {} !'.format(self.__class__.__name__,len(MIN_MATCH_COUNT),len(data.putative_matches)))

        print('[{}] Image alignment is done.'.format(self.__class__.__name__))
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
        keypoints: FAST keypoints that including coordinates
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
    putative_matches: list = None
    aligned_image: np.ndarray = None
    homography: np.ndarray = None
    # matchesMask: any = None
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


    CHECK_FOLDER = os.path.isdir(BASE_PATH)
    if not CHECK_FOLDER:
        os.makedirs(BASE_PATH)

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

    # EX3. a new processing of loading an image and extracting features
    _LoadImage_FeatureExtraction = Data('LoadImage_FeatureExtraction',[('img_path','./image/box.png'),('input_image',None),('input_keypoints',None),('input_descriptors',None)]).get_data()
    print('_LoadImage_FeatureExtraction => '+str(_LoadImage_FeatureExtraction))
    processes = [LoadImage,FeatureExtraction(maxFeatures=200,method=0)]
    vision_pipeline = PipelineBase(processes)
    _LoadImage_FeatureExtraction = vision_pipeline.execute(_LoadImage_FeatureExtraction)
    # print('_LoadImage_FeatureExtraction.input_keypoints => ')
    # Display().show_keypoints(_LoadImage_FeatureExtraction.input_keypoints)

    #EX4. A pairs of image features generated by LocalFeaturesPairs, then do feature matching and alignment
    PairData = Data('PairData',[('queryImg_path','./image/box.png'),('templateImg_path','./image/box_in_scene.png')]).get_data()
    vision_pipeline = PipelineBase([LocalFeaturesPairs(maxFeatures=200,method=0),FeatureMatching(keepPercent=0.5,method=0,filter=False),ImageAlignment])
    PairData = vision_pipeline.execute(PairData)
    # print('PairData => ')
    # Display().show_keypoints(PairData.query_keypoints)
    # Display().show_descriptors(PairData.query_descriptors)
    # Display().show_keypoints(PairData.template_keypoints)
    # Display().show_descriptors(PairData.template_descriptors)
    # Display().show_matches(PairData.matches)

    Display().draw_keypoints(PairData.template_image,PairData.template_keypoints,save_path=KPS_TEMPLATE_PATH)
    Display().draw_keypoints(PairData.query_image,PairData.query_keypoints,save_path=KPS_QUERY_PATH)
    Display().draw_matches(PairData.query_image, PairData.query_keypoints, PairData.template_image, PairData.template_keypoints, PairData.putative_matches, mode=FEATURE_MATCHING,matchesMask=None, save_path=MATCHES_PATH)
    Display().draw_matches(PairData.query_image, PairData.query_keypoints, PairData.template_image, PairData.template_keypoints, PairData.putative_matches, mode=FEATURE_MATCHING,matchesMask=PairData.matchesMask, save_path=MATCHES_PATH[:-4]+'_inliers.png')
    WriteImage().execute(PairData.aligned_image, save_path=ALIGN_PATH)

    evaluation_data = Data('evaluation_data',[('template_image',PairData.template_image),('template_keypoints',PairData.template_keypoints),('query_keypoints',PairData.query_keypoints),('homography',PairData.homography),('matches',PairData.matches),('putative_matches',PairData.putative_matches)]).get_data()
    vision_pipeline = PipelineBase([Evaluation])
    evaluation_data = vision_pipeline.execute(evaluation_data)







