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

from xkeypoint import cv2 as cv
from xkeypoint import numpy as np

from xkeypoint import KeypointDetector
from xkeypoint import KeypointDescriptor
from xkeypoint import KeypointMatcher

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
    print(f"KeypointDescriptor class usage demo started")
    MIN_MATCH_COUNT = 5
    
    # Read images
    query = cv.imread('box.png')
    train = cv.imread('box_in_scene.png')
    
    # Initiate keypoint detector and descriptor
    detector = KeypointDetector()
    descriptor = KeypointDescriptor()
    matcher = KeypointMatcher()
    
    for detector_method in ["SuperPoint"]: #["SIFT", "SURF", "ORB", "STAR", "AKAZE", "KAZE", "FAST", "BLOB", "SuperPoint"]:
        for descriptor_method in ["SuperPoint"]: #["SIFT", "SURF", "ORB", "AKAZE", "KAZE", "BRISK", "SuperPoint"]:
            if (detector_method, descriptor_method) in [
                                                        ("SIFT", "ORB"),
                                                        ("SIFT", "AKAZE"),
                                                        ("SIFT", "KAZE"),
                                                        ("SURF", "AKAZE"),
                                                        ("SURF", "KAZE"),
                                                        ("ORB", "AKAZE"),
                                                        ("ORB", "KAZE"),
                                                        ("STAR", "AKAZE"),
                                                        ("STAR", "KAZE"),
                                                        ("KAZE", "AKAZE"),
                                                        ("FAST", "ORB"),
                                                        ("FAST", "AKAZE"),
                                                        ("FAST", "KAZE"),
                                                        ("BLOB", "AKAZE"),
                                                        ("BLOB", "KAZE"),
                                                        ("SIFT", "SuperPoint"),
                                                        ("SURF", "SuperPoint"),
                                                        ("ORB", "SuperPoint"),
                                                        ("STAR", "SuperPoint"),
                                                        ("AKAZE", "SuperPoint"),
                                                        ("KAZE", "SuperPoint"),
                                                        ("FAST", "SuperPoint"),
                                                        ("BLOB", "SuperPoint"),
                                                        ("SuperPoint", "AKAZE"),
                                                        ("SuperPoint", "KAZE"),
                                                       ]:
                # Skip these pairs
                continue
            # find the keypoints
            kp1, kp1_metadata = detector(query, detector_method)
            kp2, kp2_metadata = detector(train, detector_method)
            
            # compute keypoints' descriptors
            kp1, des1, des1_metadata = descriptor(query, kp1, descriptor_method)
            kp2, des2, des2_metadata = descriptor(train, kp2, descriptor_method)
            
            metadata1 = {**kp1_metadata, **des1_metadata}
            metadata2 = {**kp2_metadata, **des2_metadata}
            
            if des1.shape[0] < 1 or des2.shape[0] < 1:
                continue

            # matching keypoints
            matches = matcher((query, train), (kp1, kp2), (des1, des2), {**metadata1, **metadata2})
            
            train_homography = train.copy()
            if len(matches)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
                 
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                 
                h,w = query.shape[:2]
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv.perspectiveTransform(pts,M)
                 
                train_homography = cv.polylines(train_homography,[np.int32(dst)],True,(255, 255, 255),3, cv.LINE_AA)
            else:
                # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
                matchesMask = None
            matchesMask = None
            draw_params = dict(
                matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2
                )
             
            canvas = cv.drawMatches(query,kp1,train_homography,kp2,matches,None,**draw_params)
            
            text = type(f"text", (), {})
            text.content = f"Detector {detector_method} <-> Descriptor {descriptor_method}"
            text.font = cv.FONT_HERSHEY_SIMPLEX
            text.scale = 0.5
            text.thickness = 1
            text.size = cv.getTextSize(
                            text.content,   # Text string
                            text.font,      # Font Family
                            text.scale,     # Font Scale
                            text.thickness  # Line Thickness in px
                            )
            (text.width, text.height), text.baseline = text.size
            text.linetype = cv.LINE_AA
            cv.putText(
                    img              = canvas,                                                  # Image to manipulate
                    text             = text.content,                                            # Text string to be written
                    org              = (0, canvas.shape[0] - 2*(text.height + text.baseline)),  # Text bottom-left corner position
                    fontFace         = text.font,                                               # Font Family
                    fontScale        = text.scale,                                              # Font Scale
                    color            = (255, 255, 255),                                         # Color in BGR
                    thickness        = text.thickness,                                          # Line Thickness in px
                    lineType         = text.linetype,                                           # Line Type
                    bottomLeftOrigin = False                                                    # Image Origin bottom-left if True, top-left otherwise
                    )
            text.content = f"Matches {len(matches)} / {MIN_MATCH_COUNT}"
            cv.putText(
                    img              = canvas,                                                  # Image to manipulate
                    text             = text.content,                                            # Text string to be written
                    org              = (0, canvas.shape[0] - (text.height + text.baseline)),    # Text bottom-left corner position
                    fontFace         = text.font,                                               # Font Family
                    fontScale        = text.scale,                                              # Font Scale
                    color            = (255, 255, 255),                                         # Color in BGR
                    thickness        = text.thickness,                                          # Line Thickness in px
                    lineType         = text.linetype,                                           # Line Type
                    bottomLeftOrigin = False                                                    # Image Origin bottom-left if True, top-left otherwise
                    )
        
            cv.imshow("canvas", canvas)
            cv.waitKey()
        
    print(f"KeypointDescriptor class usage demo completed")
    
## #############################################################################
## #### END OF FILE ############################################################
## #############################################################################