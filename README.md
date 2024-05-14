# Machine Learning: OCR for Manga

Part II of my Machine Learning exercise, utilizing manga109 curated data.  Please note that you will have to go and request for your own [Manga109 datasets](http://www.manga109.org/ja/index.html).

By the way, if you've somehow landed here looking for the "real" (proven to work and used by others) manga-ocr, hop over to [manga-ocr](https://github.com/kha-white/manga-ocr).  It's impressive, and they use "manga109-s" dataset!

In a nutshell, rather than utilizing other researcher's post-trained data, I want to go through the entire excercises of building my own.  Another thing is that I am not too fond of Python (ever since the days when I had to battle between 2.7 and 3.x and constantly getting my Gentoo broken, and vowed to avoid Python like a plague thereafter).  I will follow papers written by smarter people than I such as [this paper](https://github.com/microsoft/unilm/tree/master/trocr) from researchers of Microsoft, [transformer](https://github.com/huggingface/transformers), and [this](https://huggingface.co/docs/transformers/v4.40.2/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel), etc, but as mentioned, not the Python part :stuck_out_tongue_winking_eye: (yes, that trauma on 10+ years of Gentoo was so bad, I quit using Gentoo and switched to Debian, never looking back! - I've also learned that stability is more important than bleeding edge technology, especially on O/S which I have to use it daily!  I kept lying to myself, and telling myself "it'll get better" for 10+ years!)

## Training

In a nutshell, I have 2 phases to training, first is to train to locate WHERE the texts are, and 2nd is to OCR.  Originally, I was hoping to use out-of-the-box OCR (i.e. TensorFlowLite keras-OCR) but TensorFlow version is only English OCR.  Then there is Google ML Kit v2 OCR, which does handle Japanese, but it turns out it only supports Android and iOS.  In any case, read my experiments, trial-and-error, etc on the training directory if interested.

### Pretrained OCR's

On the side-note, if you are just interested in OCR for English, TensorFlow OCR does both phases as part of one API, it will detect the text location, and then transform the located text from pixels to text/UTF-8 (array of `char[]`s)

Tesseract is another library that can detect/locate text, and transform to text (`char[]`s) and it can (not too well, but it tries) to transform pixels to string for vertically oriented Japanese (`jpn_vert`) and horizontal (`jpn`) trained data.

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

This section is what I've learned through this educational journeys...

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