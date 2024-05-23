# Text Detection

Firstly, I want to clarify the differences as I understand it, between 'classification' and 'object detection'.  In a nutshell, 'object detection' can classify multiple objects (i.e. an image is of 2 men and 3 dogs).

Next, I'm not quite familiar with all the formats the smart people of the ML world are used to, but one format that caught my eyes were Microsoft COCO format, in which the annotations of Manga109 resembles quite strongly with.  I cannot afford to use Azure AI, but if you can afford it, look into [COCO Format Conversion](https://github.com/manga109/manga109-demos/tree/master/coco-format-conversion) tool by the smart folks from manga109.

Initially, I had started my path towards [TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) model, but it turns out they have become deprecated and the README suggested I'd either use [TensorFlow Vision](https://github.com/tensorflow/models/tree/master/official/vision) or [Google Scenic](https://github.com/google-research/scenic), and follow the the [Import COCO files from elsewhere](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/model-customization?tabs=studio#import-coco-files-from-elsewhere) section as your starting point.

After few minutes of head-scratching and dice-throwing, I've began my new paths towards TF-Vision mainly because:

- TF-Vision is developed by TensorFlow team, and because I'm too poor to afford high-performance computer for home, I have to use Google CoLab if I do not want to wait days for data to become trained, hence I lean towards any libraries that are TensorFlow friendly (integrates seamlessly with CoLab)...
- TFVision models can be fine-tuned using transfer learning, where you start with a pre-trained model and adapt it to your specific dataset. I can use the manga109 images along with their associated metadata (which consists of rectangle locations of the texts) for fine-tuning the model to detect text regions.
- Because Google CoLab comes with TensorFlow pre-installed, TFVision (which is part of TensorFlow) can be accessed without additional setup.

In the end, this approach was dropped completely after I've encountered the disk usage issue of prepping the data (I could probably reduce the image down to say 256x256 per page, but I was thinking too programmer-based method rather than data-scientist-based method, so I abandoned it completely).  The image preprocessing (I guess it's [filtering](https://learn.microsoft.com/en-us/training/modules/analyze-images-computer-vision/2-understand-computer-vision)) took a bit, and I think I should have split that into a separate work as a tool maybe (i.e. using manga109api library, but if possible in rust...)

Next, I've gone to the research on text (image) detection via Tensorflow (integrated with keras) and have discovered that most commonly practiced approaches were to use [Faster R-CNN](https://en.wikipedia.org/wiki/Object_detection) and [Single Shot MultiBox Detector (SSD)](https://en.wikipedia.org/wiki/Object_detection) (note that for robotics, I'm told YOLO is also considered), and have learnt that SSD is easier to implement (to understand and debug), so I've gone through that path.

I've learned about how N-dimensionalization of tokens (via transformers) in which objects with similar classes clusters at same area, in which I kind of now understand those 3D plots...  The spatial plotting of CNN...  This also explains that `X_Something` and `Y_Something` naming conventions in terms of plotting...

I've learned that it's an art of its own that a novice like myself should not dive in unless I'm trying to get a PhD or something...  I've learned that there are already training tools, in specific [YOLO that does object detection](https://docs.ultralytics.com/tasks/detect/).  At the time of this writing, since YOLO v8 was avaiable, I think I'll challenge myself towards that.  One attractive thing about YOLO is that YOLO can be trained both by Python and/or CLI.  Either way, as long as it can run on CoLab, I'll take either/or.

## TL;DR: Step-by-step Recipe

xxx

## Tutorials, researchs, educations, and examples

This readme MD have become just mess of thoughts from multiple days of self-educating and researches of how to approach all this.  Prior to starting this project, I only had vague comprehension of what it means to "train" an ML.  I was aware of it, but I did not realize how complex it was.  At the time of this writing, I cannot afford owning any high (or for the fact, mid) performance GPU, nor can I afford to "rent" cloud GPU to run TensorFlow, hence I'd be running trainings on Google CoLab on a single CPU mode.  It was still better than running those trainings on my local hosts.

Once I've learned about existance of HuggingFace transformer models, which generous scientists have spent days on X-GPU's (i.e. 3 days on 8 GPUs) and shared their trained models for the world to use, I've began using their data as my base model.  I'd imagine models that took 3 days at 8-GPUs, if I had to train the same thing on *my* machine, it'll probably take 5 days or longer, or worse fry my CPU (2 of my PC's fried already within past 1.5 years)...

Another thing I've learned is that I really want to use R-CNN (or YOLO) to detect where the text is, but accuracy wise, I cannot (or should not) compare the rectangles defined in Manga109 annotation meta-data with what the training determined because the rectangle coordinates can (almost always) differ by pixels.  What has to be compared (whether it trained correctly) is the OCR bit, in which whether if the annotation thinks that the text in that bounding box is "Apples and Oranges", and the OCR recognized as "apples and oranges" (it's not the `char[]` array byte-by-byte that matters, it's the accuracies of how close it has detected).

But that too has triggered a hindsight...  A manga page will have multiple regions of text boxes, in fact, in a single panel, there may be more than one text rectangle boxes of 2 (or more) persons talking, or one person speaking in both speach-bubble and thought-bubble.  Which text belongs to which?  If the OCR recognized the work "Apple" on panel 2 because it scanned from left-to-right, but on panel 1 (Japanese manga panels are right-to-left) was the text box of "Orange".  In which case, I do need to observe the rectangle coordinates in the annotation.

The thought, espcially for a ML-novice, did cross my mind, as a programmer, this thought is probably the correct solution, is to preprocess all the images, in which this tool (obviously written in Rust) will read the rectangle coordinates of the annotation, cut out the image from the mapped (matching) page, and save it onto the disk.  And basically train based on just the cut-outs.  If that was the case, I can even go further, and just run the [Tesseract tool](https://github.com/tesseract-ocr/tessdoc/blob/main/Fonts.md) which will automagically put a bounding box around each kanji/kana character of same dimensions, and I I have to do is train my own [Tesseract Data](https://tesseract-ocr.github.io/tessdoc/Data-Files.html) (`.traineddata`).

I may still lean towards that paths, especially because I've used Tesseract in the past project, and just recently also learned (while toying on the idea of training my own) that Tesseract 4 has its own [NeuralNet](https://github.com/tesseract-ocr/tessdoc/blob/main/tess4/NeuralNetsInTesseract4.00.md) engine (via usage of `-psm 13` arg).  It's quite attractive, especially because the (latest) NN supports vertical Japanese.  And to top it off, compared to other trained-model for Manga OCR, Tesseract is quite fast!  It's written in pure C++ and takes advantage of OpenMP and SSE!

On my [other project](https://github.com/HidekiAI/lenzu), I've abandoned the use of Tesseract mainly because I did not have any method to detect for text regions.  Tesseract does well almost always if there are only texts in the image.  I got the feeling that it's an OCR based on text-only images.  If I can isolate and focus (crop) just the text without any noise of manga images, it has quite high accuracy.  In fact, even without proving, I can predict that Tesseract will do quite well without the Manga109 data as long as I can indicate to Tessesract the focused text region.

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

One thing both EAST and CRAFT do well apparently is that it can detect text better when they are horizontal.  It doesn't really know what language of text, all it knows is that it's predicting that it may be some text, therefore I'd like to write a separate tool to preprocess the data, in which I'd rotate the image counter-clockwise 90degrees and have its annotation to also have the rectangle rotated 90deg to match the rotated image.  Though this would mean that whenever I request to evalute a page (on the client side application), I would need to request with 2 images, one up-right and the other rotated, and when the evaluator determines the rectangle bounding box for text, if it only detects higher prediction on the rotated one, pass that on to the next layer.

What dawned on me is that perhaps even though the documentations says it'll walk through the entire subdirectories, it may be looking at wrong book associated to annotations, hence my last attempt before abandoning SDD is to make sure that annotations book title matches the images book directory.  This sent me through another rabbit-hole for which I was going to look at the actual source code logic of what it was doing, and then I learned that the method `flow_from_directory()` is deprecated! (see my rant!)

I've switched over to Keras v2 method to load blocks of samples, and started off with very small samples of "books" (I'll come back to this emphasis on "books" later).  I need to first present the directory structures of manga109 datasets:

  ```bash
  project_root_dir/
  ├── images/
  │   ├── book1/
  │   │   ├── page1.jpg
  ...
  │   │   └── pageN.jpg
  ...
  │   └── bookN/
  │       ├── page1.jpg
  ...
  │       └── pageN.jpg
  └── annotations/
      ├── book1.xml
      ...
      └── bookN.xml
  ├── books.txt  <--------------------- this is the list of books in the dataset, i.e. `$ls images > books.txt`
  ```

So for example, if I want to test training small sets of data, for example, 3 books, I'd copy 3 dirs into 'images' directory, and 3 matching XML files from 'annotations', and then fix-up 'books.txt' to match the book directories.

I have to do the same for my test data which will be used to test the trained data against to validate the region it detected are of the correct classes.  For clarifications, there are 4 classes: "frame", "text", "body", and "face".

As a starter, I decided to have my test data of 4 "books".  When I ran my initial test with new `keras.utils.image_dataset_from_directory()` implementations, I started getting failures when I attempted to call `my_model.evaluate()` that there were catagory mismatch.  Neural network output was expecting 3 channels, but I have 4...  Weird, there's 4 classes ("frames", "text", "body", and "face"), not 3...

It turns out it `image_dataset_from_directory()` treats each "book" directories as a class.  If I had 109 manga books, that means it has 109 classes...  Go take a look at the [directory structure](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory) on the documentations, and notice that the directory names are 'class_a', 'class_b', etc.  Each dir are classes.

One (trival) approach I've thought about was to split the groups into 2 classes based on annotations.  One directory contains images without text, and another directory is with.  Teach/train the ML to classify the two.  I'm pretty sure Windows (unlike Linux) will have issues of max number of files one can have in a single folder, so I'll probably train in multiple clusters.  Filenaming will be `<bookname>_<page>.jpg` so that even when the folders are flattened, it won't overwrite/collide.  Once the data is preprocessed, I don't need the annotations folder, and let it train for that...

So in order for me to stay on track, I am listing what I need here:

- Using [Manga109API](https://github.com/manga109/manga109api), I can iterate through "books"
- Using [CRAFT Word Detection](https://github.com/clovaai/CRAFT-pytorch) to train models is prefered than to write manu rules/logic/code.
- Using models such as [CTC](https://en.wikipedia.org/wiki/Connectionist_temporal_classification) loss to classify unknown sequences characters on the image (see article link in the [Links](#links) section), but this assumes evolution in time associated to horizontal coordinate, for vertical texts, I'm thinking of reading each pixels in vertical AND from right to left...

## Links

- [Convolutional Nueral Networks CNN](https://learn.microsoft.com/en-us/training/modules/analyze-images-computer-vision/2b-computer-vision-models) - CNN-based models has been around as an object detection computer vision solutions for many years.
- [Text Classification Using Convolutional Neural Networks](https://www.youtube.com/watch?v=8YsZXTpFRO0) - probably my currently most favorite explanations of CNN, this gentleman knows how to explain (teach).
- [OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr/) - Keras example on OCR; from what I understand, Keras built-in OCR is English only.
- [Japanese OCR with the CTC Loss](https://medium.com/@natsunoyuki/ocr-with-the-ctc-loss-efa62ebd8625) by Y. Natsume and [CTC Loss OCR.ipynb](https://github.com/natsunoyuki/Data_Science/blob/master/CTC%20Loss%20OCR.ipynb)
- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/) by Awni Hannun
- [Object Detection using Yolo-v5](https://github.com/nicknochnack/StopSign) (Nick Renotte) - Do watch his [video](https://www.youtube.com/watch?v=zPWxavBy1X0), he's one of the more entertaining, smart, and knowledgeable (at least to me) gems on YouTube.

Side note: When reading these papers and articles, at first, I could not comprehend this weird conventions of naming the variables with `x_somthing` and `y_something`, for example:

  ```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

- Example From: [Keras Functional API](https://keras.io/guides/functional_api/)
  
Then it dawned on me that these crazy mathmatician scientist is thinking in terms of `y = f(x)`, where `y_something` represents output/results and `x_something` represents input/features.

In any case I won't be following this convention even if I comprehend the reason; I'm somewhat contradicting myself since I used to love [Hungarian notation](https://en.wikipedia.org/wiki/Hungarian_notation) when I used to do mostly C++.  Even in C# and F# days (F# is type-inference preferred), I sometimes sneak in that convention and was criticized like I was some moron.  Even in Rust, I sometimes do something obvious like:

  ```rust
  let str_my_var: String = String::new()
  ```

My gawd, redundancies nightmare, and it's not even correct since if it is `str_something`, shouldn't it be of type `&str`? :inocent:

In any case, these kinds of notations goes away after few years, so I now avoid the temptations and just make each variables meaningful.  Recently, I've come back to using `i` as my index variable for `for` loop in C++ and Rust (when I use `enumerate()`) but I used to hate reading somebody else's code that only used `x`, `y`, `i`, `j`, and `k`.  Incidentally, from what I was told, Fortran 77 (yes, I have coded in Fortran 77 on Dec VAX!) is where the `i`, `j`, and `k` started from, which was because it's part of the language syntax that they represented iteration variable in the loop.  It's fixed...  Just like in 6502, you only have `A` (accumulator), `X` and `Y` (mainly for indexing and offset), or in 8086, it was `AX`, `BX`, `CX`... etc...  In any case, arrogant-selfish people who use these single letter variables, unless you can convince me that you've coded in Assembly Language at least for a year WITHOUT complaining that you do not understand what the purpose of the registers are for inside disassembler (that's another way of saying debugging, but without symbols), no matter how smart you are, you suck!  (I've professionally coded in Assembly Language (at SEGA Interactive) for 7 years, and hated debugging assembly language...)

## Postmortem
