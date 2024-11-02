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
from xkeypoint import combinations

from xkeypoint import Matcher
from xkeypoint import SuperPoint
from xkeypoint.superpoint import SuperGlue as _SuperGlue

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

class SuperGlue(Matcher):
    def __new__(cls):
        # For any new instance creation, check the existance of the singleton instance
        if not hasattr(cls, "_singleton"):
            # if singleton instance does NOT exist, create one
            cls._singleton = super().__new__(cls)
            # Initialize super-glue matcher
            cls.superglue = _SuperGlue()
        # always return the singleton instance
        return cls._singleton
    
    def match(self, descriptors, keypoints, images):
        output = []
        for image_pair in combinations(iterable = zip(images, keypoints, descriptors), r = 2):
            (image1, (keypoints_method1, image_keypoints1), (descriptors_method1, image_descriptors1)),\
            (image2, (keypoints_method2, image_keypoints2), (descriptors_method2, image_descriptors2)) = image_pair

            if (descriptors_method1 in [SuperPoint.__name__] and descriptors_method2 in [SuperPoint.__name__])\
                or False:
                self._matcher = self.superglue
            else:
                raise RuntimeError(f"Un-supported descriptor methods `{descriptors_method1}` and `{descriptors_method2}`")
            # Compute matches
            output.append(self._matcher((image1, image2), (image_keypoints1, image_keypoints2), (image_descriptors1, image_descriptors2)))
            
            # store all the good matches as per Lowe's ratio test.
            matches = output[-1]
            # for match_pair in output[-1]:
            #     try:
            #         m, n = match_pair
            #     except:
            #         continue
            #     if m.distance < 0.7*n.distance:
            #         matches.append(m)
            output[-1] = (f"{self._matcher.__class__.__name__}", matches)
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