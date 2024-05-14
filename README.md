# Machine Learning: OCR for Manga

Part II of my Machine Learning exercise, utilizing manga109 curated data.  Please note that you will have to go and request for your own [Manga109 datasets](http://www.manga109.org/ja/index.html).

By the way, if you've somehow landed here looking for the "real" (proven to work and used by others) manga-ocr, hop over to [manga-ocr](https://github.com/kha-white/manga-ocr).  It's impressive, and they use "manga109-s" dataset!

1. [Two phases training](#two-phases-training)
    - [Phase 1: Text Detection](#phase-1-text-detection)
    - [Phase 2 OCR: Text Recognition](#phase-2-ocr-text-recognition)
    - [Collected Text](#collected-text)
2. [The Lib and Demo](#the-lib-and-demo)
3. [Links](#links)

In a nutshell, rather than utilizing other researcher's post-trained data, I want to go through the entire excercises of building my own.  Another thing is that I am not too fond of Python (ever since the days when I had to battle between 2.7 and 3.x and constantly getting my Gentoo broken, and vowed to avoid Python like a plague thereafter).  I will follow papers written by smarter people than I such as [this paper](https://github.com/microsoft/unilm/tree/master/trocr) from researchers of Microsoft, [transformer](https://github.com/huggingface/transformers), and [this](https://huggingface.co/docs/transformers/v4.40.2/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel), etc, but as mentioned, not the Python part :stuck_out_tongue_winking_eye: (yes, that trauma on 10+ years of Gentoo was so bad, I quit using Gentoo and switched to Debian, never looking back! - I've also learned that stability is more important than bleeding edge technology, especially on O/S which I have to use it daily!  I kept lying to myself, and telling myself "it'll get better" for 10+ years!)

## Two phases training

There will be two parts to train, the first part is the object detection part, and the second is text recognition (OCR) part.

Note that even though I rant about Python, as long as I do not have to install it manually (meaning, I don't have to have that headache of figuring out which version of Python and PIP to install, etc) only other irritations I have is that darn column-alignment (Python, F#, and COBOL) that drives me nuts when I try to pretty-format and the formatter got confused and removed or added indention and all goes to angry-land...  It's not a bad language (actually, the wealth of libraries makes it a great tool!) so at least for Jupyter/Notebook under VSCode and CoLab (browser), I'll tolerate Python for building my training models.  Whether I like it or not, for ML Training using CoLab, I submit.

Just a brief note on the question of why 2 phases?  In the first phase, we basically wish to classify whether the image has any text or not.  Manga109 has 4 classifications (at least, last I've checked): body, face, frame, and text.  For my purpose, I just need it to identify correctly where the text is, and correctly place a rectangle bounding box around it.

Once we have the bounding box area, we can then pass that focused image (rectangle) to OCR which can then decide whether that text is Japanese or not, and if it is Japanese, proceed forward into transforming the image into actual UTF-8 text.

From a lazy programmers' perspective (like myself), it would be more ideal to transform the text bounding area to UTF-8 the moment we find it.  But, another way to see it is, from optimization point of view, by first training to get as accurate as possible on determing (i.e. above 85%) that there is some text on the image, we can speed up the training concentrating just on that part.  Once we get good at identifying the text rectangle, we then can go into training how well we can transform the pixels into UTF-8 text, and get that part high on its training.  If that doesn't convince you, think of it this way, how parallel can you make it if you had both tasks in one, versus splitting workers to locate all the text, and then splitting workers to transform/OCR each rects found?

### Phase 1: Text Detection

Initially, I had started my path towards [TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) model, but it turns out they have become deprecated and the README suggested I'd either use [TensorFlow Vision](https://github.com/tensorflow/models/tree/master/official/vision) or [Google Scenic](https://github.com/google-research/scenic).

After few minutes of head-scratching and dice-throwing, I've began my new paths towards TF-Vision mainly because:

- TF-Vision is developed by TensorFlow team, and because I'm too poor to afford high-performance computer for home, I have to use Google CoLab if I do not want to wait days for data to become trained, hence I lean towards any libraries that are TensorFlow friendly (integrates seamlessly with CoLab)...
- TFVision models can be fine-tuned using transfer learning, where you start with a pre-trained model and adapt it to your specific dataset. I can use the manga109 images along with their associated metadata (which consists of rectangle locations of the texts) for fine-tuning the model to detect text regions.
- Because Google CoLab comes with TensorFlow pre-installed, TFVision (which is part of TensorFlow) can be accessed without additional setup.

In the end, this approach was dropped completely after I've encountered the disk usage issue of prepping the data (I could probably reduce the image down to say 256x256 per page, but I was thinking too programmer-based method rather than data-scientist-based method, so I abandoned it completely).

Next, I've gone to the research on text (image) detection via Tensorflow (integrated with keras) and have discovered that most commonly practiced approaches were to use [Faster R-CNN](https://en.wikipedia.org/wiki/Object_detection) and [Single Shot MultiBox Detector (SSD)](https://en.wikipedia.org/wiki/Object_detection) (note that for robotics, I'm told YOLO is also considered), and have learnt that SSD is easier to implement (to understand and debug), so I've gone through that path.



### Phase 2 OCR: Text Recognition

Once my system can identify text on each manga panels, it's time to have it evaluate whether it is Japanese or some other language via classification.  From what I understand, Tensorflow already comes (built-in) with OCR, so I'll probably be using that.  I'll discover all this and postmortem my discoveries.

TODO: discuss steps and postmortems of obstacles encountered...

### Collected Text

Once the text has been collected, from here on, it's up to the user to determine what to do with it.  What you do with it is beyond the scope of this excercise, but here are few suggestions:

Ideally for example, you'd want to send that stream of text down to [mecab](https://github.com/taku910/mecab) so that it can be broken down into meaningful data.

If you're not familiar with Japanese, you first have to realize that there are no spaces between words to break the sequences of kanji's to turn them into "words".  It's not just kanjis, one of my favorite example mecab uses is the phrase 「すもももももももものうち！」which mecab can break the words down to:

  ```bash
hidekiai@jankenpon:~/projects/ML-manga109-ocr$ echo "すもももももももものうち" | mecab
すもも  名詞,普通名詞,一般,,,,スモモ,李,すもも,スモモ,すもも,スモモ,和,"","","","","","",体,スモモ,スモモ,スモモ,スモモ,"0","C2","",15660352771596800,56972
も      助詞,係助詞,,,,,モ,も,も,モ,も,モ,和,"","","","","","",係助,モ,モ,モ,モ,"","動詞%F2@-1,形容詞%F4@-2,名詞%F1","",10324972564259328,37562
もも    名詞,普通名詞,一般,,,,モモ,桃,もも,モモ,もも,モモ,和,"","","","","","",体,モモ,モモ,モモ,モモ,"0","C3","",10425303000293888,37927
も      助詞,係助詞,,,,,モ,も,も,モ,も,モ,和,"","","","","","",係助,モ,モ,モ,モ,"","動詞%F2@-1,形容詞%F4@-2,名詞%F1","",10324972564259328,37562
もも    名詞,普通名詞,一般,,,,モモ,桃,もも,モモ,もも,モモ,和,"","","","","","",体,モモ,モモ,モモ,モモ,"0","C3","",10425303000293888,37927
の      助詞,格助詞,,,,,ノ,の,の,ノ,の,ノ,和,"","","","","","",格助,ノ,ノ,ノ,ノ,"","名詞%F1","",7968444268028416,28989
うち    名詞,普通名詞,副詞可能,,,,ウチ,内,うち,ウチ,うち,ウチ,和,"","","","","","",体,ウチ,ウチ,ウチ,ウチ,"0","C3","",881267193291264,3206
EOS
  ```
  
As you can see, once they are broken down to words (i.e. noun, verbs, etc), you can then pass each words down to dictionaries via clipboard or something, etc.

Another example may be that rather than passing it down to mecab to break it down to words, you just want to identify each charater is kanji or kana, and you do not care about the neighbor-combinations of characters to make a word or not.  In such case, you can just pass it down to libraries such as kakasi, in which it has an interface (I think, it's been a while) to check if the character is a kanji or not.

Final example may be that you just needed that raw text so that you can pass it down to text-to-speach (TTS) system to have your manga read aloud for you.

As for me, what I want to do with it are two things, first is to collect all the "phrases" from as much manga as possible, and create a database of commonly used phrases that you ONLY read in manga but never in real life, and train a chatbot with it.  The bot will have this illusion that the use of languages are phrases of the manga is the truth.  I've seen videos on YouTube in which these Japanese teachers would beg foreigners not to assume that the Japanese they've learned from watching anime is the way common Japanese speaks, and that made me chuckle when I sometimes read genius comments by Japanese audiences who'd say stuff like 「やればできる子です」which they've obviously got it from manga.  The other day, I was reading a manga (I think it was 「 陰の実力者になりたくて!」) in which the character started practicing phrases he'd wish to say one of these days (I think he said 「残像だ！」) and made me want to create a bot which only spoke in this way (one of my favorite is 「見覚えのない天井だ」).  The prhases will be (initially) publicized in a form of either CSV (raw text) in which other users can participate in adding their own phrases to the base-trained-model to make the bot more smarter.  Alternatively, it can become a simple MotD database via [fortune](https://linux.die.net/man/6/fortune) and have [Cowsay](https://linux.die.net/man/1/cowsay) just randomly [wall](https://linux.die.net/man/1/wall) via `cron.hourly` (though I probably won't `wall` it since  it's VERY ANNOYING when a `wall` message just scroll my terminal whil I'm in `vim` coding...).  

Note that I do not have to necessarily do all this work since all I have to do is just go to each of the meta-data directories in manga109 and just parse the [annotation](http://www.manga109.org/ja/annotations.html) (also [acm paper annotation](https://dl.acm.org/doi/10.1145/3011549.3011551)) meta-data (they are in XML format) in the text and dump it, and I can probably also scrape all the text from [小説家になろう](https://syosetu.com/)...  So OCR is really moot for this purpose, but ideally if I had a tool that made it useful for others to OCR quickly from their collections of manga (OK, the real truth is, I just wanted to go through this excercise to learn what it takes to train ML, what kind of training do I need to do, how to integrate TensorFlow as well as utilizing it as a tool, and how to use the trained result to make it useful, so that I can say "I did it").

The 2nd usage is of course, for my [lenzu](https://github.com/HidekiAI/lenzu).

## The Lib and Demo

The demo will just scan the image and dump the rectangle pixel-coordinates (relative to upper-left of the image as origin (0,0)) where the text was found, and the raw-text (kanji and kana as-is).  See the section on "Collected Text" for some suggestions on what to do with these raw text.

TODO: Discuss about the sample-demo app which binds the trained model for seeking texts and trained model for transforming image to text.

## Links

- [manga109s](http://www.manga109.org/ja/index.html) - you must request for download with goals/purposes/intentions of what you want to do with it.  Note that if you're browser and/or proxy is strict, you won't be able to reach this site.  I think they're using self-signed cert and MS Edge for example, refuses to let me access it (I had to actually download [tor](https://www.torproject.org/) to access it - holy kow! - I figured that out when some "is the site down" sites would report back it's up and running, while others that are more security/reliable tells me it's down)
- [manga109 tools and API](https://github.com/manga109) - make sure to use this!  You shouldn't have to waste your time writing parsers/readers/writers, nor futz with XML Schemas, etc.  Let this API do all that for you, so you can concentrate on the real work!  All in all, IF you have acquired the manga109 datasets, your preperation should include browsing all the projects 
- cc-100 [ja](https://data.statmt.org/cc-100/ja.txt.xz) - I'm not sure if I'll use this yet, but still playing around with it for now...

## Postmortems, caveats, and hindsights

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


## Citations

  ```
  @article{multimedia_aizawa_2020,
    author={Kiyoharu Aizawa and Azuma Fujimoto and Atsushi Otsubo and Toru Ogawa and Yusuke Matsui and Koki Tsubota and Hikaru Ikuta},
    title={Building a Manga Dataset ``Manga109'' with Annotations for Multimedia Applications},
    journal={IEEE MultiMedia},
    volume={27},
    number={2},
    pages={8--18},
    doi={10.1109/mmul.2020.2987895},
    year={2020}
  }
  ```