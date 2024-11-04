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
from xkeypoint import numpy as np
from xkeypoint import combinations
from xkeypoint import SIFT as KeyPointDetector
from xkeypoint import SIFT as KeyPointDescriber
from xkeypoint import FLANN as KeyPointMatcher

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
    detector = KeyPointDetector()
    describer = KeyPointDescriber()
    matcher = KeyPointMatcher()
    images = lambda : (cv2.imread(path) for path in ["img1.ppm", "img2.ppm", "img3.ppm", "img4.ppm", "img5.ppm", "img6.ppm"]) # ["box.png", "box_in_scene.png"]
    
    images_keypoints = detector.detect(images())
    images_descriptors = describer.describe(images_keypoints, images())
    images_matches = matcher.match(images_descriptors, images_keypoints, images())
    
    for index, (image, (_, keypoints)) in enumerate(zip(images(), images_keypoints)):
        image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(f"image-{index}", image)

    for image_keypoints_pair, (matches_method, matches) in zip(combinations(zip(list(images()), images_keypoints), 2), images_matches):
        MIN_MATCH_COUNT = 10
        cv = cv2
        (query, kp1), (train_homography, kp2) = image_keypoints_pair
        _, kp1 = kp1 # ignore the keypoint detection method
        _, kp2 = kp2 # ignore the keypoint detection method
    
        if len(matches)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
             
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
             
            h,w = query.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
             
            train_homography = cv.polylines(train_homography.copy(),[np.int32(dst)],True,(255, 255, 255),3, cv.LINE_AA)
        else:
            # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
        # matchesMask = None # Force drawing ALL keypoints
        draw_params = dict(
            matchColor = (0,255,0), # draw matches in green color
            singlePointColor = None,
            matchesMask = matchesMask, # draw only inliers
            flags = 2
            )
         
        canvas = cv.drawMatches(query,kp1,train_homography,kp2,matches,None,**draw_params)
        
        text = type(f"text", (), {})
        text.content = f"Detector {images_keypoints[0][0]} <-> Descriptor {images_descriptors[0][0]}"
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
    cv.destroyAllWindows()
## #############################################################################
## #### END OF FILE ############################################################
## #############################################################################