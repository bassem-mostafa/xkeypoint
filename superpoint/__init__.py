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

from .superpoint import SuperPoint
from .superglue import SuperGlue as _SuperGlue

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
#
# class SuperPoint:
#     '''
#     Wrapper of superpoint actual model
#     '''
#     def __init__(self):
#         ...
#
#     def detect(self, image):
#         height, width = image.shape[:2]
#         if height < 8 or width < 8:
#             # Invalid image dimension to be evaluated
#             keypoints = []
#         else:
#             detector = _SuperPoint({})
#
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             image = torch.from_numpy(image/255.).float()[None, None]
#             output = detector({'image': image})
#             output.update(
#                           {
#                           "keypoints": output["keypoints"][0].detach().numpy(),
#                           "scores": output["scores"][0].detach().numpy(),
#                           "descriptors": output["descriptors"][0].detach().numpy().T, # Note the `.T`
#                           }
#                          )
#             keypoints = [cv2.KeyPoint(x = keypoint[0], y = keypoint[1], size = 1, response = score) for keypoint, score, _descriptor in zip(output["keypoints"], output["scores"], output["descriptors"])]
#         return keypoints
#
#     def compute(self, image, keypoints):
#         height, width = image.shape[:2]
#         if height < 8 or width < 8:
#             # Invalid image dimension to be evaluated
#             keypoints, descriptors = [], []
#         else:
#             describer = _SuperPoint({})
#
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             image = torch.from_numpy(image/255.).float()[None, None]
#             output = describer({'image': image})
#             output.update(
#                           {
#                           "keypoints": output["keypoints"][0].detach().numpy(),
#                           "scores": output["scores"][0].detach().numpy(),
#                           "descriptors": output["descriptors"][0].detach().numpy().T, # Note the `.T`
#                           }
#                          )
#
#         try:
#             # Filter out keypoints NOT existing in `keypoints` argument
#             keypoints, descriptors = zip(*[(cv2.KeyPoint(x = keypoint[0], y = keypoint[1], size = 1, response = score), descriptor) for keypoint, score, descriptor in zip(output["keypoints"], output["scores"], output["descriptors"]) if (keypoint[0], keypoint[1]) in [keypoint.pt for keypoint in keypoints]])
#
#             # No filter
#             # keypoints, descriptors = zip(*[(cv2.KeyPoint(x = keypoint[0], y = keypoint[1], size = 1, response = 0), descriptor) for keypoint, score, descriptor in zip(output["keypoints"], output["scores"], output["descriptors"])])
#         except:
#             # In case of no keypoints to describe, return empty keypoints, desciptors
#             keypoints, descriptors = [], []
#         keypoints, descriptors = list(keypoints), cv2.numpy.array(descriptors)
#         return keypoints, descriptors

class SuperGlue:
    '''
    Wrapper of superglue actual model
    '''
    def __init__(self):
        ...

    def __call__(self, images, keypoints, descriptors):
        matcher = _SuperGlue(
                             {
                             'weights': 'indoor',
                             'match_threshold': 0.4,
                             }
                            )
        
        images = list(images)
        images[0] = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        images[0] = torch.from_numpy(images[0]/255.).float()[None, None]
        
        images[1] = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
        images[1] = torch.from_numpy(images[1]/255.).float()[None, None]
        
        output = matcher(
                          {
                          'image0': images[0], 'image1': images[1],
                          'descriptors0': torch.from_numpy(descriptors[0].T).float()[None], 'descriptors1': torch.from_numpy(descriptors[1].T).float()[None],
                          'keypoints0': torch.from_numpy(cv2.numpy.asarray([kp.pt for kp in keypoints[0]])).float()[None], 'keypoints1': torch.from_numpy(cv2.numpy.asarray([kp.pt for kp in keypoints[1]])).float()[None],
                          'scores0': torch.from_numpy(cv2.numpy.asarray([kp.response for kp in keypoints[0]])).float()[None], 'scores1': torch.from_numpy(cv2.numpy.asarray([kp.response for kp in keypoints[1]])).float()[None],
                          }
                         )
        
        # 'matches0': indices0, # use -1 for invalid match
        # 'matches1': indices1, # use -1 for invalid match
        # 'matching_scores0': mscores0,
        # 'matching_scores1': mscores1,
        
        output.update(
                      {
                      "matches0": output["matches0"][0].numpy(),
                      "matching_scores0": output["matching_scores0"][0].detach().numpy(),
                      }
                     )
        matches = [cv2.DMatch(queryIdx, trainIdx, 1.0 - output["matching_scores0"][queryIdx]) for queryIdx, trainIdx in enumerate(output["matches0"]) if trainIdx > -1]

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