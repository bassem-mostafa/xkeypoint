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
        output = self._detector.detect(tuple(images))
        output = list(output)
        for i in range(len(output)):
            # output[i] = sorted(output[-1], key = lambda kp: kp.response, reverse = True)
            # output[i] = tuple(map(lambda kp: KeyPoint(kp), output[i]))
            # Here we define the method used to detect these keypoints
            output[i] = (f"{self._detector.__class__.__name__}", output[i])
        return tuple(output)

    def describe(self, keypoints, images):
        keypoints_methods, keypoints_values = tuple(zip(*keypoints))
        for method in keypoints_methods:
            if method not in [self._describer.__class__.__name__]:
                raise RuntimeError(f"Un-supported keypoints detector `{method}`")
        output = self._describer.compute(tuple(images), keypoints_values)[1] # TODO Check if returned keypoints need to be processed
        output = list(output)
        for i in range(len(output)):
            # # Update key-points' descriptors
            # for desc, kp in zip(output[i], keypoints_values):
            #     kp.descriptor[f"{self._describer.__class__.__name__}"] = desc
            # Here we define the method used to compute these descriptors
            output[i] = (f"{self._describer.__class__.__name__}", output[i])
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