# Training

Trained models are generated in 2 phases:

1. Text Detection
2. Text Recognition

In order to train text-detection model, you first need to request and acquire Manga109 data.

Once you have the curated data shared in your Google Drive, you can then train the model to detect where it thinks the text are.

Once the model has learned to detect where the text box region/rectangle is, it can then use that isolated rectangle (with less noise) the OCR can attempt to character-recognize the concentrated region.

Note that even though I rant about Python, as long as I do not have to install it manually (meaning, I don't have to have that headache of figuring out which version of Python and PIP to install, etc) only other irritations I have is that darn column-alignment (Python, F#, and COBOL) that drives me nuts when I try to pretty-format and the formatter got confused and removed or added indention and all goes to angry-land...  It's not a bad language (actually, the wealth of libraries makes it a great tool!) so at least for Jupyter/Notebook under VSCode and CoLab (browser), I'll tolerate Python for building my training models.  Whether I like it or not, for ML Training using CoLab, I submit.

Just a brief note on the question of why 2 phases?  In the first phase, we basically wish to classify whether the image has any text or not.  Manga109 has 4 classifications (at least, last I've checked): body, face, frame, and text.  For my purpose, I just need it to identify correctly where the text is, and correctly place a rectangle bounding box around it.

Once we have the bounding box area, we can then pass that focused image (rectangle) to OCR which can then decide whether that text is Japanese or not, and if it is Japanese, proceed forward into transforming the image into actual UTF-8 text.

From a lazy programmers' perspective (like myself), it would be more ideal to transform the text bounding area to UTF-8 the moment we find it.  But, another way to see it is, from optimization point of view, by first training to get as accurate as possible on determing (i.e. above 85%) that there is some text on the image, we can speed up the training concentrating just on that part.  Once we get good at identifying the text rectangle, we then can go into training how well we can transform the pixels into UTF-8 text, and get that part high on its training.  If that doesn't convince you, think of it this way, how parallel can you make it if you had both tasks in one, versus splitting workers to locate all the text, and then splitting workers to transform/OCR each rects found?

## Phase 1: Text Detection

Initially, I had started my path towards [TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) model, but it turns out they have become deprecated and the README suggested I'd either use [TensorFlow Vision](https://github.com/tensorflow/models/tree/master/official/vision) or [Google Scenic](https://github.com/google-research/scenic).

After few minutes of head-scratching and dice-throwing, I've began my new paths towards TF-Vision mainly because:

- TF-Vision is developed by TensorFlow team, and because I'm too poor to afford high-performance computer for home, I have to use Google CoLab if I do not want to wait days for data to become trained, hence I lean towards any libraries that are TensorFlow friendly (integrates seamlessly with CoLab)...
- TFVision models can be fine-tuned using transfer learning, where you start with a pre-trained model and adapt it to your specific dataset. I can use the manga109 images along with their associated metadata (which consists of rectangle locations of the texts) for fine-tuning the model to detect text regions.
- Because Google CoLab comes with TensorFlow pre-installed, TFVision (which is part of TensorFlow) can be accessed without additional setup.

In the end, this approach was dropped completely after I've encountered the disk usage issue of prepping the data (I could probably reduce the image down to say 256x256 per page, but I was thinking too programmer-based method rather than data-scientist-based method, so I abandoned it completely).

Next, I've gone to the research on text (image) detection via Tensorflow (integrated with keras) and have discovered that most commonly practiced approaches were to use [Faster R-CNN](https://en.wikipedia.org/wiki/Object_detection) and [Single Shot MultiBox Detector (SSD)](https://en.wikipedia.org/wiki/Object_detection) (note that for robotics, I'm told YOLO is also considered), and have learnt that SSD is easier to implement (to understand and debug), so I've gone through that path.



## Phase 2 OCR: Text Recognition

Once my system can identify text on each manga panels, it's time to have it evaluate whether it is Japanese or some other language via classification.  From what I understand, Tensorflow already comes (built-in) with OCR, so I'll probably be using that.  I'll discover all this and postmortem my discoveries.

TODO: discuss steps and postmortems of obstacles encountered...


## Postmortem, hindsights, and caveats

Skip paragraph below, it's just my rant:

  First of all, before I go into all of it, I just want to rant on how much I passionately dispise how poorly Jupiter Notebook integrates with vim.  When you naviate (cursor mode) within the cells, if you hit 'k' or 'j' once too many, your cursor goes outside the cell and it goes into somewhere unpredictable.  Am I exaggerating?  You can only understand this if you've actually experienced it...  All I can say is that it's a real big pain coding and debugging, and as previously mentioned, because I dislike Python, it's a double whammy!  Maybe for a data-scientists, they're OK with this, but for a non-Python programmer (C++, Rust, C#, etc) dealing with this is probably something that requires tolerances in high degree.

End ranting...

The very first issue I've encountered was during training of image/text detection.  The entire collections of manga images are 3+ gigs compressed.  Trying to load the entire set of images (multiple books) is going to place a toll on RAM; worse was that I was preprocessing (i.e. `keras.preprocessing.image.img_to_array()` and `eras.application.efficientnet.preprocess_input()`) each pages of multiple books and saving it as `TFRecord`.  My goals were to attempt to load the partial (batch) blocks of TFRecord records into RAM (via `shuffle()` method), and attempt on several epochs.

I didn't even get that far, I got stuck on running out of disk space building TFRecord records for each images.  If you're curious, my record was as follows:

  ```python
  def create_tf_manga109_rects_from_page(preprocessed_image, page_width, page_height, text_rects):
    # Initialize lists for rectangle coordinates
    xmin_list, ymin_list, xmax_list, ymax_list = [], [], [], []

    for text_rect in text_rects:
        # Append rectangle coordinates to the lists
        xmin_list.append(text_rect['@xmin'])
        ymin_list.append(text_rect['@ymin'])
        xmax_list.append(text_rect['@xmax'])
        ymax_list.append(text_rect['@ymax'])

    # Create a TensorFlow Example from the image and the text regions
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[preprocessed_image.tobytes()])),
        'page_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[page_width])),
        'page_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[page_height])),
        'xmin': tf.train.Feature(int64_list=tf.train.Int64List(value=xmin_list)),
        'ymin': tf.train.Feature(int64_list=tf.train.Int64List(value=ymin_list)),
        'xmax': tf.train.Feature(int64_list=tf.train.Int64List(value=xmax_list)),
        'ymax': tf.train.Feature(int64_list=tf.train.Int64List(value=ymax_list)),
    }))

    return tf_example
  ```

Overall, I've completely abandoned this method and decided to work with TensorFlow and Keras, and after a brief research, I've wound up on whether to go with SSD or Faster R-CNN for image detections integrating with Tensorflow.  In the end, I've began my paths towards SSD mainly because it is supposed to be easier of the two due to requiring only a "single stage" (single shot).
