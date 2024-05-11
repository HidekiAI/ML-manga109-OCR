# Machine Learning: OCR for Manga

Part II of my Machine Learning exercise, utilizing manga109 curated data.  Please note that you will have to go and request for your own [Manga109 datasets](http://www.manga109.org/ja/index.html).

By the way, if you've somehow landed here looking for the "real" (proven to work and used by others) manga-ocr, hop over to [manga-ocr](https://github.com/kha-white/manga-ocr).  It's impressive, and they use "manga109-s" dataset!

In a nutshell, rather than utilizing other researcher's post-trained data, I want to go through the entire excercises of building my own.  Another thing is that I am not too fond of Python (ever since the days when I had to battle between 2.7 and 3.x and constantly getting my Gentoo broken, and vowed to avoid Python like a plague thereafter).  I will follow papers written by smarter people than I such as [this paper](https://github.com/microsoft/unilm/tree/master/trocr) from researchers of Microsoft, [transformer](https://github.com/huggingface/transformers), and [this](https://huggingface.co/docs/transformers/v4.40.2/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel), etc, but as mentioned, not the Python part :stuck_out_tongue_winking_eye: (yes, that trauma on 10+ years of Gentoo was so bad, I quit using Gentoo and switched to Debian, never looking back! - I've also learned that stability is more important than bleeding edge technology, especially on O/S which I have to use it daily!  I kept lying to myself, and telling myself "it'll get better" for 10+ years!)

## Two parts training

There will be two parts to train, the first part is the object detection part, and the second is text recognition (OCR) part.

Note that even though I rant about Python, as long as I do not have to install it manually (meaning, I don't have to have that headache of figuring out which version of Python and PIP to install, etc) only other irritations I have is that darn column-alignment (Python, F#, and COBOL) that drives me nuts when I try to pretty-format and the formatter got confused and removed or added indention and all goes to angry-land...  It's not a bad language (actually, the wealth of libraries makes it a great tool!) so at least for Jupyter/Notebook under VSCode and CoLab (browser), I'll tolerate Python for building my training models.  Whether I like it or not, for ML Training using CoLab, I submit.

### Text Detection

Initially, I had started my path towards [TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) model, but it turns out they have become deprecated and the README suggested I'd either use [TensorFlow Vision](https://github.com/tensorflow/models/tree/master/official/vision) or [Google Scenic](https://github.com/google-research/scenic).

After few minutes of head-scratching and dice-throwing, I've began my new paths towards TF-Vision mainly because:

- TF-Vision is developed by TensorFlow team, and because I'm too poor to afford high-performance computer for home, I have to use Google CoLab if I do not want to wait days for data to become trained, hence I lean towards any libraries that are TensorFlow friendly (integrates seamlessly with CoLab)...
- TFVision models can be fine-tuned using transfer learning, where you start with a pre-trained model and adapt it to your specific dataset. I can use the manga109 images along with their associated metadata (which consists of rectangle locations of the texts) for fine-tuning the model to detect text regions.
- Because Google CoLab comes with TensorFlow pre-installed, TFVision (which is part of TensorFlow) can be accessed without additional setup.

TODO: discuss steps and postmortems of obstacles encountered...

### OCR: Text Recognition

Once my system can identify text on each manga panels, it's time to have it evaluate whether it is Japanese or some other language.

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

## The Lib and Demo

The demo will just scan the image and dump the rectangle pixel-coordinates (relative to upper-left of the image as origin (0,0)) where the text was found, and the raw-text (kanji and kana as-is).  See the section on "Collected Text" for some suggestions on what to do with these raw text.

TODO: Discuss about the sample-demo app which binds the trained model for seeking texts and trained model for transforming image to text.

## Links

- [manga109s](http://www.manga109.org/ja/index.html)
- cc-100 [ja](https://data.statmt.org/cc-100/ja.txt.xz)
