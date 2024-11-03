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

class KeypointDescriptor:
    def __init__(self):
        ...
    
    def __call__(self, image, keypoints, method="SIFT"):
        '''
        Describe an image keypoints using the specified method
        
        args:
            image: input image to describe keypoints within
            keypoints: keypoints detected within input image
            method: algorithm or model to be used for keypoint describing
            
        returns:
            tuple of keypoints, descriptors, metadata
        '''
        descriptors = []
        metadata = {}
        if method in ["SIFT"]:
            describer = cv2.SIFT_create()
            keypoints, descriptors = describer.compute(image, keypoints)
            metadata["descriptors"] = "SIFT"
        elif method in ["SURF"]:
            describer = cv2.xfeatures2d.SURF_create()
            keypoints, descriptors = describer.compute(image, keypoints)
            metadata["descriptors"] = "SURF"
        elif method in ["ORB"]:
            describer = cv2.ORB_create()
            keypoints, descriptors = describer.compute(image, keypoints)
            metadata["descriptors"] = "ORB"
        elif method in ["AKAZE"]:
            describer = cv2.AKAZE_create()
            keypoints, descriptors = describer.compute(image, keypoints)
            metadata["descriptors"] = "AKAZE"
        elif method in ["KAZE"]:
            describer = cv2.KAZE_create()
            keypoints, descriptors = describer.compute(image, keypoints)
            metadata["descriptors"] = "KAZE"
        elif method in ["BRISK"]:
            describer = cv2.BRISK_create()
            keypoints, descriptors = describer.compute(image, keypoints)
            metadata["descriptors"] = "BRISK"
        elif method in ["SuperPoint"]:
            describer = SuperPoint()
            descriptors = describer.describe((keypoints), (image))
            metadata["descriptors"] = "SuperPoint"
        elif method in ["Alike"]:
            ... # TODO
        else:
            raise RuntimeError(f"Keypoint descriptor `{method}` Not Implemented")
        return keypoints, descriptors, metadata

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