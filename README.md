# Task
Following code provide solution for task of counting object 
(associated with bounding boxes from detection and track_id from tracking) appearence in ROI 
which itself described by bounding box from detector (aka dynamic ROI).

I am sure code still can by optimized despite that it is written in vectorized form (Numpy). Because of 
vectorization feature maximal number of "dynamic ROI" per one frame should be defined.

By tracking I assume presence of entities of frame_id, track_id, class_id in the data.

# Task example
Let say you have tracking of baby turtles on some video, and you what to count how many frames baby
turtles spend on mamas turtles, which themselves moving. It could work not only for single class of baby turtles
but, e.g. some little bird as well. It this case class_id have {mama_turtle, baby_turtle, little_bird}

