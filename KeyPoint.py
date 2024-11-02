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
from xkeypoint import StringIO

## #############################################################################
## #### Private Type(s) ########################################################
## #############################################################################

CVKeyPoint = cv2.KeyPoint

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

class KeyPoint(CVKeyPoint):
    def __init__(self, *args):
        # Initializes the OpenCV Key-Point attributes' values
        super().__init__()

        # Initializes the Added attributes' values
        self.point = None # stores that keypoint coordinates (x, y)
        self.descriptor = {} # stores various descriptors for that same keypoint
        
        # Overwrite/Update the default values if provided with an initialization argument
        if len(args) == 1:
            if type(args[0]) == CVKeyPoint:
                # OpenCV Key-Point deep-copy
                kp = args[0]
                self.pt = kp.pt
                self.response = kp.response
                self.size = kp.size
                self.angle = kp.angle
                self.octave = kp.octave
                self.class_id = kp.class_id
                
                # Update Added attributes values
                self.point = self.pt
            else:
                raise RuntimeError(f"Un-supported type of argument `{type(args[0])}`")
        elif len(args) > 1:
            raise RuntimeError(f"Un supported number of arguments `{len(args)}`")
    
    def __repr__(self):
        '''
        Describes the Key-Point
        returns:
            a string that describes the Key-Point
        '''
        text = StringIO()
        if False: # For debugging purposes, Enable for extra information
            print(f"{self.__class__.__name__} Attributes:", file=text)
            for key, value in self.__dict__.items():
                print(f"...\t{str(key):20s}: `{value}`", file=text)
        else:
            print(f"< {self.__class__.__name__} {id(self):016X}>", file=text, end="")
        return text.getvalue()
        

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