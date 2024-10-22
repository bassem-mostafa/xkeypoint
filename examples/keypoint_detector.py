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
from xkeypoint import KeypointDetector

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
    print(f"KeypointDetector class usage demo started")
    
    query = cv2.imread("box.png")
    train = cv2.imread("box_in_scene.png")
    canvas = cv2.numpy.zeros((max(query.shape[0], train.shape[0]), (query.shape[1] + train.shape[1]), 3), dtype=cv2.numpy.uint8)

    detector = KeypointDetector()
    
    for method in ["SIFT", "SURF", "ORB", "STAR", "AKAZE", "KAZE", "FAST", "BLOB", "SuperPoint"]:
        query_keypoints = detector(query, method)[0]
        train_keypoints = detector(train, method)[0]

        query_output = cv2.drawKeypoints(query, query_keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        train_output = cv2.drawKeypoints(train, train_keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        canvas[:] = 0
        canvas[:query.shape[0], :query.shape[1], :] = query_output
        canvas[:train.shape[0], query.shape[1]:, :] = train_output
        
        text = type(f"text", (), {})
        text.content = f"Method {method}"
        text.font = cv2.FONT_HERSHEY_SIMPLEX
        text.scale = 0.5
        text.thickness = 1
        text.size = cv2.getTextSize(
                        text.content,   # Text string
                        text.font,      # Font Family
                        text.scale,     # Font Scale
                        text.thickness  # Line Thickness in px
                        )
        (text.width, text.height), text.baseline = text.size
        text.linetype = cv2.LINE_AA
        cv2.putText(
                img              = canvas,                                                  # Image to manipulate
                text             = text.content,                                            # Text string to be written
                org              = (0, train.shape[0]-text.height),                                        # Text bottom-left corner position
                fontFace         = text.font,                                               # Font Family
                fontScale        = text.scale,                                              # Font Scale
                color            = (255, 255, 255),                                         # Color in BGR
                thickness        = text.thickness,                                          # Line Thickness in px
                lineType         = text.linetype,                                           # Line Type
                bottomLeftOrigin = False                                                    # Image Origin bottom-left if True, top-left otherwise
                )
        
        cv2.imshow("canvas", canvas)
        cv2.waitKey()
        
    print(f"KeypointDetector class usage demo completed")
    
## #############################################################################
## #### END OF FILE ############################################################
## #############################################################################