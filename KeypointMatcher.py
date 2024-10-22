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
from xkeypoint import torch
from xkeypoint.superpoint import SuperGlue

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

class KeypointMatcher:
    def __init__(self):
        ...
    
    def __call__(self, images, keypoints, descriptors, metadata={}, method=None):
        '''
        Matches images keypoints' descriptors using the specified method
        
        args:
            images: tuple of input images to match keypoints within
            keypoints: tuple of keypoints detected within input images
            descriptors: tuple of keypoints' descriptors
            metadata: tuple of keypoints' metadata
            method: algorithm or model to be used for keypoint matching
            
        returns:
            tuple of matches, metadata
        '''
        
        # TODO verifications
        
        if method is None:
            if metadata.get("descriptors", None) in ["SIFT", "SURF", "KAZE"]:
                method = ("FLANN", "KDTREE")
            elif metadata.get("descriptors", None) in ["ORB", "AKAZE", "BRISK"]:
                method = ("FLANN", "LSH")
            elif metadata.get("keypoints", None) in ["SuperPoint"] and metadata.get("descriptors", None) in ["SuperPoint"]:
                method = ("SuperGlue")
            else:
                raise RuntimeError(f"Couldn't determine method to match keypoints `{metadata.get('keypoints', None)}` with descriptors `{metadata.get('descriptors', None)}` !")
        
        if method == ("FLANN", "KDTREE"):
            search_params = dict(checks = 50)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(
                               algorithm = FLANN_INDEX_KDTREE,
                               trees = 5,
                               )
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors[0], descriptors[1], k = 2)
        elif method == ("FLANN", "LSH"):
            search_params = dict(checks = 50)
            FLANN_INDEX_LSH = 6
            index_params= dict(
                              algorithm = FLANN_INDEX_LSH,
                              table_number = 6,         # default: 12
                              key_size = 12,            # default: 20
                              multi_probe_level = 1,    # default: 2
                              )
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors[0], descriptors[1], k = 2)
        elif method == ("SuperGlue"):
            matcher = SuperGlue()
            matches = matcher(images, keypoints, descriptors)
        else:
            raise RuntimeError(f"Keypoint matcher `{method}` Not Implemented")
        
        if method[0] in ["FLANN"]:
            # store all the good matches as per Lowe's ratio test.
            good_matches = []
            for pair in matches:
                try:
                    m, n = pair
                except:
                    continue
                if m.distance < 0.7*n.distance:
                    good_matches.append(m)
        else:
            good_matches = matches
        matches = good_matches
        return matches

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