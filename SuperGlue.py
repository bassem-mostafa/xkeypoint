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
from xkeypoint import combinations

from xkeypoint import Matcher
from xkeypoint import SuperPoint

from .superpoint import SuperGlue as _SuperGlue

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
            cls._singleton.superglue = _SuperGlue(
                                                  {
                                                  'weights': 'indoor',
                                                  'match_threshold': 0.4,
                                                  }
                                                 ).cuda()
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

            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image1 = torch.from_numpy(image1/255.).float()[None, None]
            
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            image2 = torch.from_numpy(image2/255.).float()[None, None]
            
            # Compute matches
            output.append(
                         self.superglue(
                                        {
                                        'image0': image1.cuda(), 'image1': image2.cuda(),
                                        'descriptors0': torch.from_numpy(image_descriptors1.T).float()[None].cuda(), 'descriptors1': torch.from_numpy(image_descriptors2.T).float()[None].cuda(),
                                        'keypoints0': torch.from_numpy(cv2.numpy.asarray([kp.pt for kp in image_keypoints1])).float()[None].cuda(), 'keypoints1': torch.from_numpy(cv2.numpy.asarray([kp.pt for kp in image_keypoints2])).float()[None].cuda(),
                                        'scores0': torch.from_numpy(cv2.numpy.asarray([kp.response for kp in image_keypoints1])).float()[None].cuda(), 'scores1': torch.from_numpy(cv2.numpy.asarray([kp.response for kp in image_keypoints2])).float()[None].cuda(),
                                        }
                                       )
                         )
        
            # 'matches0': indices0, # use -1 for invalid match
            # 'matches1': indices1, # use -1 for invalid match
            # 'matching_scores0': mscores0,
            # 'matching_scores1': mscores1,
            
            # convert output Tensors to numpy arrays
            output[-1].update(
                              {
                              "matches0": output[-1]["matches0"][0].cpu().numpy(),
                              "matching_scores0": output[-1]["matching_scores0"][0].detach().cpu().numpy(),
                              }
                             )
            
            # convert super-glue matches to CVMatches
            output[-1] = [cv2.DMatch(queryIdx, trainIdx, 1.0 - output[-1]["matching_scores0"][queryIdx]) for queryIdx, trainIdx in enumerate(output[-1]["matches0"]) if trainIdx > -1]

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