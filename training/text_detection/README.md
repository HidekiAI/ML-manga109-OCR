
 - SDD (Single Shot MultiBox Detector)
 - EAST (Efficient and Accurate Scene Text Detector)
 - CRAFT (Character-Region Awareness for Text Detection)

 Of the three, SDD seems to be the easiest to implement, but as you can see on vertial text, our accuracy is quite poor:

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

On the other hand, EAST and CRAFT both seems to do well on vertial text, so I will try EAST next (CRAFT apparently is multi-step and so it's a bit more complex) after I try one more thing.

One thing both EAST and CRAFT do well apparently is that it can detect text better when they are horizontal.  It doesn't really know what language of text, all it knows is that it's predicting that it may be some text, therefore I'd like to write a separate tool to preprocess the data, in which I'd rotate the image counter-clockwise 90degrees and have its annotation to also have the rectangle rotated 90deg to match the rotated image.  Though this would mean that whenever I request to evalute a page (on the client side application), I would need to request with 2 images, one up-right and the other rotated;  The evalutor will then make a prediction of text bounding box on both images, and pass down the rectangle coordinate (and the image) of the one that has higher predictions to the OCR.  The OCR will then also need to be trained to not just evalute horizontal and vertial Japanese, but also rotated Japanese, so that when it evaluates the rotated version it can OCR correctly.