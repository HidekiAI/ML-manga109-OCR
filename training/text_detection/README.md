# Text Detection

Initially, I had started my path towards [TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) model, but it turns out they have become deprecated and the README suggested I'd either use [TensorFlow Vision](https://github.com/tensorflow/models/tree/master/official/vision) or [Google Scenic](https://github.com/google-research/scenic).

After few minutes of head-scratching and dice-throwing, I've began my new paths towards TF-Vision mainly because:

- TF-Vision is developed by TensorFlow team, and because I'm too poor to afford high-performance computer for home, I have to use Google CoLab if I do not want to wait days for data to become trained, hence I lean towards any libraries that are TensorFlow friendly (integrates seamlessly with CoLab)...
- TFVision models can be fine-tuned using transfer learning, where you start with a pre-trained model and adapt it to your specific dataset. I can use the manga109 images along with their associated metadata (which consists of rectangle locations of the texts) for fine-tuning the model to detect text regions.
- Because Google CoLab comes with TensorFlow pre-installed, TFVision (which is part of TensorFlow) can be accessed without additional setup.

In the end, this approach was dropped completely after I've encountered the disk usage issue of prepping the data (I could probably reduce the image down to say 256x256 per page, but I was thinking too programmer-based method rather than data-scientist-based method, so I abandoned it completely).

Next, I've gone to the research on text (image) detection via Tensorflow (integrated with keras) and have discovered that most commonly practiced approaches were to use [Faster R-CNN](https://en.wikipedia.org/wiki/Object_detection) and [Single Shot MultiBox Detector (SSD)](https://en.wikipedia.org/wiki/Object_detection) (note that for robotics, I'm told YOLO is also considered), and have learnt that SSD is easier to implement (to understand and debug), so I've gone through that path.

## Choice of Models

There were few choices I've read on, and here are the 3 I've reduced down to trying:

- SDD (Single Shot MultiBox Detector)
- EAST (Efficient and Accurate Scene Text Detector)
- CRAFT (Character-Region Awareness for Text Detection)

Of the three, SDD seems to be the easiest to implement, but when I tried it out, here's the accuracy I got back:

  ```bash
  Found 6519 images belonging to 83 classes.
  Found 1587 images belonging to 83 classes.
  Epoch 1/10
  204/204 [==============================] - ETA: 0s - loss: 4.4155 - accuracy: 0.0291 
  Epoch 1: val_loss improved from inf to 4.27837, saving model to /content/drive/MyDrive/projects/ML-manga-ocr-rust/data/tf_model/
  204/204 [==============================] - 2637s 13s/step - loss: 4.4155 - accuracy: 0.0291 - val_loss: 4.2784 - val_accuracy: 0.0202
  ...
  Epoch 9/10
  204/204 [==============================] - ETA: 0s - loss: 3.5498 - accuracy: 0.1571
  Epoch 9: val_loss improved from 3.81475 to 3.78763, saving model to /content/drive/MyDrive/projects/ML-manga-ocr-rust/data/tf_model/
  204/204 [==============================] - 425s 2s/step - loss: 3.5498 - accuracy: 0.1571 - val_loss: 3.7876 - val_accuracy: 0.1342
  Epoch 10/10
  204/204 [==============================] - ETA: 0s - loss: 3.5092 - accuracy: 0.1671
  Epoch 10: val_loss improved from 3.78763 to 3.71605, saving model to /content/drive/MyDrive/projects/ML-manga-ocr-rust/data/tf_model/
  204/204 [==============================] - 415s 2s/step - loss: 3.5092 - accuracy: 0.1671 - val_loss: 3.7160 - val_accuracy: 0.1285
  ```

Each batch took an average of 500s or so, having total of 10 results to 5000 seconds (about 80 minutes).  One thing I've learned was that unlike local Jupyter, as long as my browser stays open, even if my browser goes to idle/sleep, Google CoLab will continue where it left off!  Of course, just like Jupyter if you crash while processing hours of data, you're dead if it could not save the result weights...

The accuracy is just crummy, I'm thinking it's SDD is very poor on verticle texts...  On the other hand, EAST and CRAFT both seems to do better on vertial text, so I will try EAST next (CRAFT apparently is multi-step and so it's a bit more complex) after I try one more thing.

One thing both EAST and CRAFT do well apparently is that it can detect text better when they are horizontal.  It doesn't really know what language of text, all it knows is that it's predicting that it may be some text, therefore I'd like to write a separate tool to preprocess the data, in which I'd rotate the image counter-clockwise 90degrees and have its annotation to also have the rectangle rotated 90deg to match the rotated image.  Though this would mean that whenever I request to evalute a page (on the client side application), I would need to request with 2 images, one up-right and the other rotated, and when the evaluator determines the rectangle bounding box for text, if it only detects 

What dawned on me is that perhaps even though the documentations says it'll walk through the entire subdirectories, it may be looking at wrong book associated to annotations, hence my last attempt before abandoning SDD is to make sure that annotations book title matches the images book directory.
