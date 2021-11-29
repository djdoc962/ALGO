import cv2
import numpy as np



class ImageAlignment():

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

    def match(self, descripA, descripB,keepPercent=0.2, method=0):
        """
        match the features between images
        matches: 
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

    def Match2Keypoint(self,matches,KptA,KptB):
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

    def find_homography(self,kpsA, kpsB, matches, method=0):
        """
        calculate homography matrix (perspective transformation), should have at least 4 corresponding point
        H: homography matrix
        """
        if method == 0:
            method = cv2.RANSAC

        ptsA, ptsB = self.Match2Keypoint(matches,kpsA,kpsB)
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
        return H

    def wrap(self, image, H, size ):
        """
        align image via transformation
        wraped: wraped image
        """
        # if( len(image.shape) > 2 ):
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        wraped = cv2.warpPerspective(image, H, size)
        return wraped

    def show_keypoints(self,keypoints):
        print('keypoints length: {} ======================================'.format(len(keypoints)))
        for i, keypoint in enumerate(keypoints):
            print('i: {}, (x,y): ({}, {})'.format(i,keypoint.pt[0],keypoint.pt[1]))

    def show_descriptors(self,descriptors):
        print('descriptors length: {} with dimension: {} ======================================'.format(len(descriptors),len(descriptors[0])))
        for i, feature in  enumerate(descriptors):
            print('i: {}, feature: {}'.format(i,feature[:]))

    def show_matches(self,matches):
        print('matches length: {} ======================================'.format(len(matches)))
        for i, match in enumerate(matches):
            print('i: {}, Idx: {}, {}'.format(i,match.queryIdx,match.trainIdx))