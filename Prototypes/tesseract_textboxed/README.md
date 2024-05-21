# Prototype: Tesseract Boxed Text

This experiment is to prove how well (accurate) Tesseract can OCR texts.

One of the classifications for Manga109 is the "text" in each "pages" of the "books".

What this tool does is the following:

1. Read the annotations specified and make a cutout if regions (rectangle coordinates) indicated by the XML "text" elements.
2. Once the text region has been saved to a RAM disk (to have least impact on I/O), it will OCR that (whole) image.
3. It will then compare what was OCR'd and what was indicated in the "text" element char-by-char (UTF-8) and determine how many characters it matched (or missed).
4. Report the difference; maybe just report "book", "page", "text_id", "char_count", "matched_count".  Note that the text_id is an unique ID defined in the annotation file.

Based on the CSV report generated, I can just see it on the spreadsheet of how accurate Tesseract is...

## Result so far...

Initially, I've tried `-psm 13` in hopes that I can take advantages of NN but it was disappointing.  I've switched back to `-psm 5` mainly because that works the most well on `jpn_vert`.

I've tried different methods of comparing accuracies.  As a starter, I've tried to do compared based on `starts_with()` method, mainly because the OCR may possibly add unwanted carriage-return/linefeed/line-breaks:

```rust
// to avoid diff in with/without LF, we'll just do "starts_with"
let match_expected = ocr_result.starts_with(expected);
println!("Expected: '{}' ({} chars)", expected, expected.len());
println!("OCR Result: '{}' ({} chars)", ocr_result, ocr_result.len());
println!("Matched: {}", match_expected);
```

```csv
book.title,page,text.id,text.characters.count,match_starts_with,matched_character_count,missed_character_count,missed_characters.as_array()
"ARMS","2","00000001","3","true","1","2",""
"ARMS","3","00000005","12","false","0","12","キ,ャ,ー,ッ"
"ARMS","3","00000007","51","false","0","51","は,や,く,逃,げ,な,い,と,ま,き,ぞ,え,く,っ,ち,ゃ,う"
"ARMS","3","0000000b","6","false","0","6","え,？"
"ARMS","3","00000013","54","false","1","53","の,時,代,は,人,を,見,た,ら,敵,と,思,え,な,の,よ,っ"
"ARMS","3","00000015","33","false","2","31","ち,は,あ,ぶ,な,い,よ,…,…"
"ARMS","4","00000032","58","false","0","58","あ,の,変,態,男,も,今,度,こ,そ,振,り,切,れ,た,は,ず,.,.,.,.,.,.,."
"ARMS","4","00000034","6","false","1","5","あ"
"ARMS","4","00000037","85","false","4","81","に,し,て,も,し,つ,こ,い,な,い,つ,ま,で,私,に,つ,き,ま,と,う,も,り,か,し,ら"
"ARMS","4","0000003b","81","false","1","80","う,1,0,日,ぐ,ら,い,に,は,な,る,ん,じ,ゃ,.,.,.,.,.,.,.,
,1,ヵ,月,ぐ,ら,い,だ,っ,け,.,.,.,.,.,.,?"
"ARMS","4","0000003c","6","false","0","6","は,あ"
"ARMS","4","0000003e","6","false","1","5","あ"
```

Firstly, just as an indication that byte-count for UTF-8 varies per character.  For example, a character "あ" will count for 3 bytes.

In any case, even with this approach, I have to conclude (for now) Tesseract "jpn_vert" using `-psm 5` (note that `-psm 13` is disappointment) for now is inaccurate even when the image has been isolated to focused bounding box.

## When I have time...

Possibly, if I get ambitious, I may possibly attempt to compare expected-vs-result by first removing spaces, commas, carriage-returns, etc so that I am only comparing accuracies based on sequential order of appearances of kana and kanji.  But then again, stripping away "?", "!", or even "..." means something if my attempt to OCR was to convert to speech (TTS).  In such case, expressions matter, hence OCR fails if it misses the punctuations.  Again, it won't matter if I am just trying to get accuracies based on sequences of characters recognized.

Another note, is that OCR (especially Japanese) will always have difficulties with differentiating 「つ」 versus 「っ」.  For a novice who do not read Japanese, s/he will just argue that it's just a font-size difference, but it is of same chararacter.  OCR is a novice.  It only knows what it is taught,  and OCR such as Tesseract, which is generalized OCR that attempts to handle all languages will stumble on per-language specific rules.

Incidentally, the oldest Japanese OCR had a rule in which it assumed all characters are of same height (bounding boxed), hence if you get characters 「っつ」 side-by-side, it will treat them as 2 different characters because it assumes both are bounded on same height bounding box...

Another issues with OCR's are the [diacritics](https://en.wikipedia.org/wiki/Diacritic) such as 「゛」 and 「゜」 which depending on what the DPI was, it gets confused, i.e.  「ふぶぷフブプ」.

And lastly, characters that looks alike, such as 「ノンシソ」 for kanas, and of course, kanjis such as 「愛」vs「変」, 「王」vs「玉」, 「人」vs「入」, and 「猫」vs「描」.  From UTF-8 (text) point of view, when you compare bytes-to-bytes they are different.  But from OCR point of view, imagine trying to squint your eyes, and maybe the image is a bit blurred, or maybe during the Conv2D layer, the sillouettes of 「猫」 and 「描」 looks almost the same...  

Depending on ML methods, NN can possibly learn to inspect neighbor characters, and realize that it meant to say 「猫」instead of 「描」.  But what if the context of the dialogue was about drawing a cat?  Or a cat is drawing?

I think some techniques that were also used was to lookup against dictionary/[jisho](https://github.com/neologd/mecab-ipadic-neologd) files.  In which, it will look for the kanji in the jisho, and common jisho will usually provide example of its usage, and so there will be (sequentially) neighbor characters, in which ML can guess based on similar looking character also exists.

There is a [HuggingFace model](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v2) which I speculate did just that by using [mecab](https://taku910.github.io/mecab/) to do some intelligent guessing of neighbor characters.  Mecab is very nice idea/usage because by having it break down into meaningful sequence of characters, if that chacter is between two kanji, it can look to see if its previous and next characters are similar, and if so, it can increase its predictions to higher accuracy...

## Citation

```text
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
