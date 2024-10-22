# X Key-point

X Key-point is a unified key-point interface for multiple key-point detectors, descriptors, and matchers.
# What is a key-point ?

Key-point is a unique point of interest that has special specifications. _A simple example of a key-point is **corner points**_.

![Corner Point](<Harris Corners.png>)

![Checkerboard Inner Corners](<checkerboard inner corners.png>)
# What does key-points used for ?

As key-points identifies and localizes **points of interest** in an image. It has been utilized in various computer vision tasks. Most known task is key-points detection, _also known as key-point localization or landmark detection_, which provides an essential information about location, pose, and structure of objects or entities within an image, playing a critical role in computer vision applications such as these:
- Pose estimation.
- Object detection and tracking.
- Facial analysis.
- Augmented reality.

<table>
<td><img src="pose%20estimation.jpg" alt="Human Pose" height=400px width=400px/></td>
<td><img src="facial%20detection.png" alt="Facial analysis" height=400px width=400px/></td>
</table>

# Building blocks

## `Keypoint`

`Keypoint` class provides a unified interface for key-points.

```Python
class Keypoint:
	...
```
## `KeypointDetector`

`KeypointDetector` class provides a unified interface for key-point detectors.

```Python
class KeypointDetector:
	...
```
## `KeypointDescriptor`

`KeypointDescriptor` class provides a unified interface for key-point descriptors.

```Python
class KeypointDescriptor:
	...
```
## `KeypointMatcher`

`KeypointMatcher` class provides a unified interface for key-point matchers.

```Python
class KeypointMatcher:
	...
```
