#here I would like to use the simple mediapipe post extraction to bound clips around different parts of the body
#the aim is that I could extract feet positions, and do so for many different clips - ballet, salsa, walking, random etc and try to run these as concepts
#I could build this up with time, including arm position, body proximity, to find different dancing concepts

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = '/Users/fleurconway/Documents/Programming/Kinetics_400_First_Test/pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)


#If you use the video mode or live stream mode, Pose Landmarker uses tracking to avoid triggering the model on every frame, which helps reduce latency.
with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.