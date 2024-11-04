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
from xkeypoint import SIFT

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

class FLANN(Matcher):
    def __new__(cls):
        # For any new instance creation, check the existance of the singleton instance
        if not hasattr(FLANN, "_singleton"):
            # if singleton instance does NOT exist, create one
            FLANN._singleton = super(FLANN, cls).__new__(cls)
            # Initialize flann-kdtree matcher
            search_params = dict(checks = 50)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(
                               algorithm = FLANN_INDEX_KDTREE,
                               trees = 5,
                               )
            FLANN._singleton.kdtree = cv2.FlannBasedMatcher(index_params, search_params)
            # Initialize flann-lsh matcher
            search_params = dict(checks = 50)
            FLANN_INDEX_LSH = 6
            index_params= dict(
                              algorithm = FLANN_INDEX_LSH,
                              table_number = 6,         # default: 12
                              key_size = 12,            # default: 20
                              multi_probe_level = 1,    # default: 2
                              )
            FLANN._singleton.lsh = cv2.FlannBasedMatcher(index_params, search_params)
        # always return the singleton instance
        return FLANN._singleton
    
    def match(self, descriptors, keypoints, images):
        output = []
        for image_pair in combinations(iterable = zip(images, keypoints, descriptors), r = 2):
            (image1, (keypoints_method1, image_keypoints1), (descriptors_method1, image_descriptors1)),\
            (image2, (keypoints_method2, image_keypoints2), (descriptors_method2, image_descriptors2)) = image_pair

            if (descriptors_method1 in [SIFT.__name__] and descriptors_method2 in [SIFT.__name__])\
                or False:
                self._matcher = self.kdtree
            else:
                raise RuntimeError(f"Un-supported descriptor methods `{descriptors_method1}` and `{descriptors_method2}`")
            # Compute matches
            output.append(self._matcher.knnMatch(image_descriptors1, image_descriptors2, k = 2))
            
            # store all the good matches as per Lowe's ratio test.
            matches = []
            for match_pair in output[-1]:
                try:
                    m, n = match_pair
                except:
                    continue
                if m.distance < 0.7*n.distance:
                    matches.append(m)
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