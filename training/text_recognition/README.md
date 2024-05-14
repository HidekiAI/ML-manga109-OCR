# Optical Character Recognition

It turns out Tensoflow (Lite) [OCR](https://www.tensorflow.org/lite/examples/optical_character_recognition/overview) are broken down into 2 stages, in which first is the text-detection, and second is text recognition.  So out of box, you *can* use OCR via TFLite *IF* you only need to recognize English!

On the other hand, [Google ML Kit](https://developers.google.com/ml-kit/vision/text-recognition) can [recognize Japanese](https://developers.google.com/ml-kit/vision/text-recognition/v2/languages) *for v2*.

Then comes the sad part...  The ML Kit only is for Android and iOS.  Hence, in the end, I have two choices:

1. Use [Tesseeract](https://github.com/tesseract-ocr/tesseract) - we can avoid the training completely and just trust it, but from my experiences from my [other project](https://github.com/HidekiAI/lenzu), Japanese text (horizontal and vertial) did not do too well.
2. CRNN model via TensorFlow/Keras - Using Convolutional Recurrent Neural Network based on annotation dataset


## Vertical Text

Top-to-bottom-left-to-right 

One suggestion is to rotate the rectangle in -90 degrees (counter-clockwise 90deg) and train OCR if it detects that it's not horizontal.  From ML's point of view, it should not matter if it is taught that vertically and rotated-to-horizontal both maps to same text, it'll always get the text, predict it, then rotate it, and predict again, and if they are of similar prediction, we call it a match.


## Links

- [Tensorflow OCR](https://www.tensorflow.org/lite/examples/optical_character_recognition/overview)
- [ML Kit Text Recognition v2](https://developers.google.com/ml-kit/vision/text-recognition/v2)

It