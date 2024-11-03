## #############################################################################
## #### Copyright ##############################################################
## #############################################################################

'''
Copyright 2024 BaSSeM

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

## #############################################################################
## #### Description ############################################################
## #############################################################################

## #############################################################################
## #### Control Variable(s) ####################################################
## #############################################################################

## #############################################################################
## #### Import(s) ##############################################################
## #############################################################################

from xkeypoint import cv2
from xkeypoint import SuperPoint

## #############################################################################
## #### Private Type(s) ########################################################
## #############################################################################

## #############################################################################
## #### Private Method(s) Prototype ############################################
## #############################################################################

## #############################################################################
## #### Private Variable(s) ####################################################
## #############################################################################

## #############################################################################
## #### Private Method(s) ######################################################
## #############################################################################

## #############################################################################
## #### Public Method(s) Prototype #############################################
## #############################################################################

## #############################################################################
## #### Public Type(s) #########################################################
## #############################################################################

class KeypointDetector:
    def __init__(self):
        ...
    
    def __call__(self, image, method="SIFT"):
        '''
        Detects an image keypoints using the specified method
        
        args:
            image: input image to detect keypoints within
            method: algorithm or model to be used for keypoint detection
            
        returns:
            tuple of keypoints, metadata
        '''
        keypoints = []
        metadata = {}
        if method in ["SIFT"]:
            detector = cv2.SIFT_create()
            output = detector.detect(image)
            keypoints = sorted(output, key = lambda kp: kp.response, reverse = True)
            metadata["keypoints"] = "SIFT"
        elif method in ["SURF"]:
            detector = cv2.xfeatures2d.SURF_create()
            output = detector.detect(image)
            keypoints = sorted(output, key = lambda kp: kp.response, reverse = True)
            metadata["keypoints"] = "SURF"
        elif method in ["ORB"]:
            detector = cv2.ORB_create()
            output = detector.detect(image)
            keypoints = sorted(output, key = lambda kp: kp.response, reverse = True)
            metadata["keypoints"] = "ORB"
        elif method in ["STAR", "CenSurE"]:
            detector = cv2.xfeatures2d.StarDetector_create()
            output = detector.detect(image)
            keypoints = sorted(output, key = lambda kp: kp.response, reverse = True)
            metadata["keypoints"] = "STAR"
        elif method in ["AKAZE"]:
            detector = cv2.AKAZE_create()
            output = detector.detect(image)
            keypoints = sorted(output, key = lambda kp: kp.response, reverse = True)
            metadata["keypoints"] = "AKAZE"
        elif method in ["KAZE"]:
            detector = cv2.KAZE_create()
            output = detector.detect(image)
            keypoints = sorted(output, key = lambda kp: kp.response, reverse = True)
            metadata["keypoints"] = "KAZE"
        elif method in ["FAST"]:
            detector = cv2.FastFeatureDetector_create()
            output = detector.detect(image)
            keypoints = sorted(output, key = lambda kp: kp.response, reverse = True)
            metadata["keypoints"] = "FAST"
        elif method in ["BLOB"]:
            detector = cv2.SimpleBlobDetector_create()
            output = detector.detect(image)
            keypoints = sorted(output, key = lambda kp: kp.response, reverse = True)
            metadata["keypoints"] = "BLOB"
        elif method in ["SuperPoint"]:
            detector = SuperPoint()
            keypoints = detector.detect([image])[0][1]
            metadata["keypoints"] = "SuperPoint"
        elif method in ["Alike"]:
            ... # TODO
        else:
            raise RuntimeError(f"Keypoint detector `{method}` Not Implemented")

        return tuple(keypoints), metadata

## #############################################################################
## #### Public Method(s) #######################################################
## #############################################################################

## #############################################################################
## #### Public Variable(s) #####################################################
## #############################################################################

## #############################################################################
## #### Main ###################################################################
## #############################################################################

if __name__ == "__main__":
    ...
    
## #############################################################################
## #### END OF FILE ############################################################
## #############################################################################