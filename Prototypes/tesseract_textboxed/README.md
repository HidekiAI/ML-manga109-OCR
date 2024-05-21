# Prototype: Tesseract Boxed Text

This experiment is to prove how well (accurate) Tesseract can OCR texts.

One of the classifications for Manga109 is the "text" in each "pages" of the "books".

What this tool does is the following:

1. Read the annotations specified and make a cutout if regions (rectangle coordinates) indicated by the XML "text" elements.
2. Once the text region has been saved to a RAM disk (to have least impact on I/O), it will OCR that (whole) image.
3. It will then compare what was OCR'd and what was indicated in the "text" element char-by-char (UTF-8) and determine how many characters it matched (or missed).
4. Report the difference; maybe just report "book", "page", "text_id", "char_count", "matched_count".  Note that the text_id is an unique ID defined in the annotation file.

Based on the CSV report generated, I can just see it on the spreadsheet of how accurate Tesseract is...
