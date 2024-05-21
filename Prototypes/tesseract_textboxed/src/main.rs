use std::{fs::File, io::Write, path};

use manga109api;
// Why rusty_tesseract rather than tesseract?  Simple, because of the documentation.
use rusty_tesseract::Args;

fn init_ocr() -> rusty_tesseract::Args {
    //tesseract version
    let tesseract_version = rusty_tesseract::get_tesseract_version().unwrap();
    println!("Tesseract - Version is: {:?}", tesseract_version);

    //available languages
    let tesseract_langs = rusty_tesseract::get_tesseract_langs().unwrap();
    println!(
        "Tesseract - The available languages are: {:?}",
        tesseract_langs
    );

    //available config parameters
    let parameters = rusty_tesseract::get_tesseract_config_parameters().unwrap();
    println!(
        "Tesseract - Config parameter: {}",
        parameters.config_parameters.first().unwrap()
    );
    Args {
        lang: "jpn_vert+jpn+osd".into(),
        //psm: Some(13), // PSM=13 utilizes neural nets
        psm: Some(5), // PSM=5 seems to be the best we can do that is closest on jpn_vert
        ..rusty_tesseract::Args::default()
    }
}

fn main() {
    let ocr_args = init_ocr();
    let temp_image_paths = "./temp/ocr_rect.jpg";
    let report_csv = "./temp/ocr_report.csv";
    let manga109_root = "../../data/Manga109s/Manga109s_released_2023_12_07/";
    let manga109 = manga109api::Manga109::new(manga109_root);

    // overwrite existing CSV
    let mut csv_file: File = File::create(report_csv).expect("Failed to create CSV file");
    // write the header
    csv_file.write_all(
        b"book.title,page,text.id,text.characters.count,match_starts_with,matched_character_count,missed_character_count,missed_characters.as_array()\n",
    ).expect("Failed to write header to CSV file");

    // For each book's page, locate each text box and create a temp image, then run tesseract on it
    // and CSV output the results in format of:
    //      book.title, page, text.id, text.characters.count, matched_character_count, missed_character_count, missed_characters.as_array()
    for book in &manga109.books {
        // iterate through each pages
        for page in &book.pages {
            // skip if this page has no text
            if page.texts.len() == 0 {
                continue;
            }
            // page has text, it's worth it to load the image (whole page)
            let image_paths = manga109
                .img_path(book.title.as_str(), &page.index)
                .expect("Unable to get image path for {book.title}/{page.index}.jpg");
            let image_page = rusty_tesseract::image::open(image_paths)
                .expect("Unable to open image '{book.title}/{page.index}.jpg'");

            // iterate through each text box
            for text in &page.texts {
                // create a temp image of the text box
                let sub_image =
                    image_page.crop_imm(text.xmin, text.ymin, text.get_width(), text.get_height());
                // save it into a temp file so we can open it
                sub_image
                    .save(path::Path::new(temp_image_paths))
                    .expect("Failed to save temp image");

                // run tesseract on the text box
                let temp_image = rusty_tesseract::image::open(temp_image_paths)
                    .expect("Failed to run tesseract on text box");
                let rusty_image = rusty_tesseract::Image::from_dynamic_image(&temp_image)
                    .expect("Failed to convert image");
                let ocr_result = rusty_tesseract::image_to_string(&rusty_image, &ocr_args)
                    .expect("Failed to run tesseract on text box");

                // compare the result with the ground truth
                compare_ocr(&mut csv_file, &book.title, page.index, &text, &ocr_result);
            }
        }
    }
    csv_file.flush().expect("Failed to flush CSV file");
}

// Note that it will get overly complicated trying to figure out if it missed a charater or two
// and attempting to align it.  For example, if the expected is ABCD and result is ACDC, it could
// maybe consider that the letter C and letter D were so close, that it was a misread.  For now
// it'll cause mismatch of 3 characters [{A, true}, {B, false}, {C, false}, {D, false}].
// Real example:
//      Expected: 'あのばか
//      ウォーマシンにひっかかりやがった！' (64 chars)
//      OCR Result: 'あの ば ぱか
//      ウォ ー マ シ ン に
//      ひび ひっかかり
//      
//      や が っ た /。 .
//      ' (97 chars)
//      Matched: false
//      
//      Expected: '超重子弾か！？' (21 chars)
//      OCR Result: '超重 子 弾
//      さっ 1
//      ' (26 chars)
//      Matched: false
fn compare_ocr(
    writer: &mut File,
    title: &String,
    index: usize,
    text: &manga109api::Text,
    ocr_result: &String,
) {
    let expected = &text.value;

    // to avoid diff in with/without LF, we'll just do "starts_with"
    let match_expected = ocr_result.starts_with(expected);
    println!("Expected: '{}' ({} chars)", expected, expected.len());
    println!("OCR Result: '{}' ({} chars)", ocr_result, ocr_result.len());
    println!("Matched: {}", match_expected);

    let count_diff = (expected.len() as i32 - ocr_result.len() as i32).abs();

    // NOTE: zip() will prematurely stop at the shortest of the two iterators
    //let zip_diff = expected
    //    .chars()
    //    .zip(ocr_result.chars())
    //    .filter(|(a, b)| a != b);
    let from_expected = expected
        .clone()
        .chars()
        .enumerate()
        .map(|(i, c)| {
            let found = ocr_result.contains(c); // use this to determine what OCR could not recognize
            let imatch = ocr_result.chars().nth(i).unwrap_or(' ') == c;
            (c, imatch, found)
        })
        .collect::<Vec<(char, bool, bool)>>();
    // build a list of characters that were not in the expected result due to OCR misprediction
    let not_in_expected = ocr_result
        .chars()
        .filter_map(|c| if !expected.contains(c) { Some(c) } else { None })
        .collect::<Vec<char>>();

    // book.title, page, text.id, text.characters.count, matched_character_count, missed_character_count, missed_characters.as_array()
    let matched_characters_count = from_expected
        .iter()
        .filter_map(|(c, imatch, found)| if *imatch { Some(c) } else { None }) // I want only the ones where imatch == true
        .count();
    let missed_character_count = expected.len() - matched_characters_count;
    let missed_chars_csv = from_expected
        .iter()
        .filter_map(|(c, imatch, found)| if !imatch { Some(c) } else { None })
        .map(|c| c.to_string())
        .collect::<Vec<String>>()
        .join(",");

    // NOTE: explicitly adds line breaks, so use print!() instead of println!()
    // there are no spaces to separate the values, so it's easier to read in a CSV viewer
    // in which only delimiter is a comma, and the column is always wrapped in double quotes
    let csv_out = format!(
        "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n",
        title,
        index,
        text.id,
        expected.len(),           // expected character count
        match_expected,           // matched based on starts_with()
        matched_characters_count, // matched character count
        missed_character_count,   // missed character count
        missed_chars_csv          // missed characters
    );
    //print!("{}", csv_out.replace(",", ", "));
    writer
        .write_all(csv_out.as_bytes())
        .expect("Failed to write to CSV file");
}
