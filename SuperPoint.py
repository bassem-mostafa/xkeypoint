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

from xkeypoint import Detector
from xkeypoint import Describer

from .superpoint import SuperPoint as _SuperPoint

## #############################################################################
## #### Private Type(s) ########################################################
## #############################################################################

## #############################################################################
## #### Private Method(s) Prototype ############################################
## #############################################################################

## #############################################################################
## #### Private Variable(s) ####################################################
## #############################################################################

_device = 'cuda' if torch.cuda.is_available() else 'cpu'

## #############################################################################
## #### Private Method(s) ######################################################
## #############################################################################

## #############################################################################
## #### Public Method(s) Prototype #############################################
## #############################################################################

## #############################################################################
## #### Public Type(s) #########################################################
## #############################################################################

class SuperPoint(Detector, Describer):
    def __new__(cls):
        # For any new instance creation, check the existance of the singleton instance
        if not hasattr(SuperPoint, "_singleton"):
            # if singleton instance does NOT exist, create one
            SuperPoint._singleton = _SuperPoint({})
            SuperPoint._singleton.eval()
            SuperPoint._singleton.to(_device)
        # Create this cls instance, and initilize its detector and descriptor to singleton instance
        instance = super().__new__(cls)
        instance._detector = SuperPoint._singleton
        instance._describer = SuperPoint._singleton
        instance._detector.eval()
        instance._describer.eval()
        instance._detector.to(_device)
        instance._describer.to(_device)
        # return the created cls instance
        return instance
    
    def detect(self, images):
        # Convert images to tensors
        images = list(images)
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            images[i] = torch.from_numpy(images[i]/255.).float()[None].to(_device)
        images = torch.stack(images)
        
        output = self._detector({'image': images})
        # convert output Tensors to numpy arrays
        output.update(
                      {
                      "keypoints": [elem.detach().cpu().numpy() for elem in output["keypoints"]],
                      "scores": [elem.detach().cpu().numpy() for elem in output["scores"]],
                      "descriptors": [elem.detach().cpu().numpy() for elem in output["descriptors"]],
                      }
                     )
        # convert the keypoints to CVKeypoints
        output.update(
                      {
                      "keypoints": [[cv2.KeyPoint(x = keypoint[0], y = keypoint[1], size = 1, response = score) for keypoint, score in zip(keypoints, scores)] for keypoints, scores in zip(output["keypoints"], output["scores"])],
                      }
                     )
        output = output["keypoints"]
        for i in range(len(output)):
            output[i] = (f"{self._detector.__class__.__name__}", output[i])
        return tuple(output)

    def describe(self, keypoints, images):
        keypoints_methods, keypoints_values = tuple(zip(*keypoints))
        for method in keypoints_methods:
            if method not in [self._describer.__class__.__name__]:
                raise RuntimeError(f"Un-supported keypoints detector `{method}`")

        # Convert images to tensors
        images = list(images)
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            images[i] = torch.from_numpy(images[i]/255.).float()[None].to(_device)
        images = torch.stack(images)
        
        output = self._detector({'image': images})
        # convert output Tensors to numpy arrays
        output.update(
                      {
                      "keypoints": [elem.detach().cpu().numpy() for elem in output["keypoints"]],
                      "scores": [elem.detach().cpu().numpy() for elem in output["scores"]],
                      "descriptors": [elem.detach().cpu().numpy() for elem in output["descriptors"]],
                      }
                     )
        # convert the keypoints to CVKeypoints
        output.update(
                      {
                      "keypoints": [[cv2.KeyPoint(x = keypoint[0], y = keypoint[1], size = 1, response = score) for keypoint, score in zip(keypoints, scores)] for keypoints, scores in zip(output["keypoints"], output["scores"])],
                      }
                     )
        
        output = output["descriptors"]
        for i in range(len(output)):
            output[i] = (f"{self._detector.__class__.__name__}", output[i])
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