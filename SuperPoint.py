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
from xkeypoint import numpy

from xkeypoint import KeyPoint
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
        if not hasattr(cls, "_singleton"):
            # if singleton instance does NOT exist, create one, and initialize it
            cls._singleton = super().__new__(cls)
            cls._singleton._detector = _SuperPoint({}).cuda()
            cls._singleton._describer = cls._singleton._detector
        # always return the singleton instance
        return cls._singleton
    
    def detect(self, images):
        output = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = torch.from_numpy(image/255.).float()[None, None].cuda()
            output.append(self._detector({'image': image}))
            # convert output Tensors to numpy arrays
            output[-1].update(
                              {
                              "keypoints": output[-1]["keypoints"][0].detach().cpu().numpy(),
                              "scores": output[-1]["scores"][0].detach().cpu().numpy(),
                              "descriptors": output[-1]["descriptors"][0].detach().cpu().numpy().T, # Note the `.T`
                              }
                             )
            # convert the keypoints to CVKeypoints then to our KeyPoint
            output[-1].update(
                              {
                              "keypoints": tuple(map(lambda kp: KeyPoint(kp), [cv2.KeyPoint(x = keypoint[0], y = keypoint[1], size = 1, response = score) for keypoint, score in zip(output[-1]["keypoints"], output[-1]["scores"])]))
                              }
                             )
            # Update key-points' descriptors
            for desc, kp in zip(output[-1]["descriptors"], output[-1]["keypoints"]):
                kp.descriptor[f"{self._describer.__class__.__name__}"] = desc
            
            # Sort keypoints with respect to relevance
            output[-1].update(
                              {
                                  "keypoints": sorted(output[-1]["keypoints"], key = lambda kp: kp.response, reverse = True),
                              }
                             )

            # Here we define the method used to detect these keypoints
            output[-1] = (f"{self._detector.__class__.__name__}", output[-1]["keypoints"])
        return tuple(output)
    
    def describe(self, keypoints, images):
        output = []
        for image, image_keypoints in zip(images, keypoints):
            keypoints_method, keypoints_values = image_keypoints
            if keypoints_method not in [self._describer.__class__.__name__]:
                raise RuntimeError(f"Un-supported keypoints detector `{keypoints_method}`")

            if False:
                # FIXME
                # TODO check the existance of the descriptor for that keypoint
                # TODO if not exist, detect and compute the descriptors
                output.append(self.detect([image])[0][1])
            else:
                output.append(keypoints_values)
            
            # retrieve the descriptors related to detected keypoints
            output[-1] = numpy.asarray(list(keypoint.descriptor[f"{self._describer.__class__.__name__}"] for keypoint in output[-1] if keypoint in keypoints_values))

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