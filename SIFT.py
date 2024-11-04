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

from xkeypoint import KeyPoint
from xkeypoint import Detector
from xkeypoint import Describer

## #############################################################################
## #### Private Type(s) ########################################################
## #############################################################################

## #############################################################################
## #### Private Method(s) Prototype ############################################
## #############################################################################

_SIFT_create = cv2.SIFT_create # Alias for OpenCV SIFT_create

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

class SIFT(Detector, Describer):
    def __new__(cls):
        # For any new instance creation, check the existance of the singleton instance
        if not hasattr(SIFT, "_singleton"):
            # if singleton instance does NOT exist, create one, and initialize it
            SIFT._singleton = super(SIFT, cls).__new__(cls)
            SIFT._singleton._detector = _SIFT_create()
            SIFT._singleton._describer = SIFT._singleton._detector
        # always return the singleton instance
        return SIFT._singleton
    
    def detect(self, images):
        output = []
        for image in images:
            output.append(self._detector.detect(image))
            output[-1] = sorted(output[-1], key = lambda kp: kp.response, reverse = True)
            output[-1] = tuple(map(lambda kp: KeyPoint(kp), output[-1]))
            # Here we define the method used to detect these keypoints
            output[-1] = (f"{self._detector.__class__.__name__}", output[-1])
        return tuple(output)

    def describe(self, keypoints, images):
        output = []
        for image, image_keypoints in zip(images, keypoints):
            keypoints_method, keypoints_values = image_keypoints
            if keypoints_method not in [self._describer.__class__.__name__]:
                raise RuntimeError(f"Un-supported keypoints detector `{keypoints_method}`")
            output.append(self._describer.compute(image, keypoints_values)[1])
            # Update key-points' descriptors
            for desc, kp in zip(output[-1], keypoints_values):
                kp.descriptor[f"{self._describer.__class__.__name__}"] = desc
            # Here we define the method used to compute these descriptors
            output[-1] = (f"{self._describer.__class__.__name__}", output[-1])
        return tuple(output)

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