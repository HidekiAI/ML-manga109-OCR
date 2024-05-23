use rand::prelude::*;
use std::{io::Write, path};

enum DatasetType {
    Train,
    Val,
}

fn normalize_paths(path: &std::path::PathBuf) -> std::path::PathBuf {
    let str_path = path
        .to_str()
        .unwrap()
        .replace("\\", "/")
        .replace("//", "/");
    std::path::PathBuf::from(path::Path::new(str_path.as_str()))
}

// Coordinates are absolute image coordinates found in Manga109 annotations
// Manga109 deals with rectangles that are assumes that origin of the image
// is at upper-left corner of the image, making all rectangles have positive
// coordinates.
// The YOLO xywh format is normalized to range [0.0 .. 1.0] (note it's positive only)
fn convert_boundingbox_to_yolo_xywh(
    image_width: u32, // we don't care whether this dimension is of original image or resized image!
    image_height: u32,
    rect_minx: u32,
    rect_miny: u32,
    rect_maxx: u32,
    rect_maxy: u32,
) -> (
    f32, /*center_x*/
    f32, /*center_y*/
    f32, /*width*/
    f32, /*height*/
) {
    if image_width == 0 || image_height == 0 {
        panic!("Image width and height must be positive");
    }
    if rect_minx >= rect_maxx || rect_miny >= rect_maxy {
        panic!("Invalid bounding box");
    }

    let scale_x = 1.0 / image_width as f32;
    let scale_y = 1.0 / image_height as f32;
    let center_x = (rect_minx + rect_maxx) as f32 / 2.0;
    let center_y = (rect_miny + rect_maxy) as f32 / 2.0;
    let width = (rect_maxx - rect_minx) as f32;
    let height = (rect_maxy - rect_miny) as f32;
    let yolo_center_x = center_x * scale_x;
    let yolo_center_y = center_y * scale_y;
    let yolo_width = width * scale_x;
    let yolo_height = height * scale_y;

    if yolo_center_x < 0.0 || yolo_center_x > 1.0 || yolo_center_y < 0.0 || yolo_center_y > 1.0 {
        panic!("Invalid YOLO center x/y");
    }
    if yolo_width < 0.0 || yolo_width > 1.0 || yolo_height < 0.0 || yolo_height > 1.0 {
        panic!("Invalid YOLO width/height");
    }
    (yolo_center_x, yolo_center_y, yolo_width, yolo_height)
}

fn images_train_dir(root_data_dir: &str) -> std::path::PathBuf {
    std::path::Path::new(root_data_dir).join("images/train")
}
fn images_val_dir(root_data_dir: &str) -> std::path::PathBuf {
    std::path::Path::new(root_data_dir).join("images/val")
}
fn labels_train_dir(root_data_dir: &str) -> std::path::PathBuf {
    std::path::Path::new(root_data_dir).join("labels/train")
}
fn labels_val_dir(root_data_dir: &str) -> std::path::PathBuf {
    std::path::Path::new(root_data_dir).join("labels/val")
}
fn mk_dataset_dir(root_data_dir: &str) {
    // 'mkdir -p' equivalent, if paths already exists, it will not panic
    std::fs::create_dir_all(images_train_dir(root_data_dir)).unwrap();
    std::fs::create_dir_all(images_val_dir(root_data_dir)).unwrap();
    std::fs::create_dir_all(labels_train_dir(root_data_dir)).unwrap();
    std::fs::create_dir_all(labels_val_dir(root_data_dir)).unwrap();
}

fn copy_image_to_dataset(
    src_book: &manga109api::Book,
    transformed_file_rootdir: &str,
    dstype: DatasetType,
    page: &usize,
) {
    //let src_annotation_file = std::path::Path::new(src_book.get_annotation_file_paths().as_str()); // i.e. annotations/title1.xml
    let img_dir_paths = src_book.get_image_dir_paths();
    let src_image_dir = std::path::Path::new(img_dir_paths.as_str()); // i.e.  images/title1/
    let src_image_path = src_image_dir.join(format!("{:03}.jpg", page)); // i.e. 009.jpg
    if src_image_path.exists() == false {
        panic!("Source Image file not found: '{:?}'", src_image_path);
    }

    let dest_image_dir = match dstype {
        DatasetType::Train => images_train_dir(transformed_file_rootdir),
        DatasetType::Val => images_val_dir(transformed_file_rootdir),
    };
    let dest_label_dir = match dstype {
        DatasetType::Train => labels_train_dir(transformed_file_rootdir),
        DatasetType::Val => labels_val_dir(transformed_file_rootdir),
    };
    let dest_image_path = dest_image_dir.join(format!("{}_{}.jpg", src_book.title.clone(), page));
    //let dest_label_path = dest_label_dir.join(format!("{}_{}.txt", title, page));

    // if dest image exists, we don't need to copy it again
    if dest_image_path.exists() == false {
        println!("Copying image from '{:?}' to '{:?}'", src_image_path, dest_image_path);
        std::fs::copy(src_image_path, dest_image_path).unwrap();
    }
}

fn get_labels_file_paths(
    transformed_file_rootdir: &str,
    dstype: DatasetType,
    title: &str,
) -> std::path::PathBuf {
    let path = match dstype {
        DatasetType::Train => labels_train_dir(transformed_file_rootdir),
        DatasetType::Val => labels_val_dir(transformed_file_rootdir),
    };
    let ret_path = path.join(format!("{}.txt", title));
    normalize_paths(&ret_path)
}

// space seprated values: class_index center_x center_y width height
fn write_yolo_label_file(
    dest_file: &mut std::fs::File,
    yolo_center_x: f32,
    yolo_center_y: f32,
    yolo_width: f32,
    yolo_height: f32,
) {
    let line = format!(
        "0 {} {} {} {}\n",
        yolo_center_x, yolo_center_y, yolo_width, yolo_height
    );
    dest_file.write_all(line.as_bytes()).unwrap();
}

fn main() {
    let training_ratio = 0.8; // 80% training, 20% validation
    let transformed_file_rootdir = "./data/";
    let manga109_root = "../../../../data/Manga109s/Manga109s_released_2023_12_07/";
    let manga109 = manga109api::Manga109::new(manga109_root);
    mk_dataset_dir(transformed_file_rootdir);

    // first, get number of books we have, and decide how many books to put in training and validation
    let num_books = manga109.books.len();
    let num_train_books = (num_books as f32 * training_ratio).round() as usize;
    //let num_val_books = num_books - num_train_books;
    // shuffle book indices to randomly select books for training and validation
    let mut book_indices: Vec<usize> = (0..num_books).collect();
    book_indices.shuffle(&mut rand::thread_rng());
    let train_books_indices = &book_indices[0..num_train_books];
    //let val_books_indices = &book_indices[num_train_books..num_books];

    // iterate through each books while converting annotations to YOLO format
    for (book_index, book) in manga109.books.iter().enumerate() {
        let is_training_dataset = train_books_indices.contains(&book_index);
        // writer based on whether book_index is in train_books_indices or val_books_indices
        let mut writer = if is_training_dataset {
            let path = get_labels_file_paths(
                transformed_file_rootdir,
                DatasetType::Train,
                &book.title.clone(),
            );
            println!("Writing to file: '{:?}'", path);
            std::fs::File::create(path).unwrap()
        } else {
            let path = get_labels_file_paths(
                transformed_file_rootdir,
                DatasetType::Val,
                &book.title.clone(),
            );
            println!("Writing to file: '{:?}'", path);
            std::fs::File::create(path).unwrap()
        };

        for page in &book.pages {
            // skip pages that has no TEXT
            if page.texts.len() == 0 {
                continue;
            }
            if is_training_dataset {
                copy_image_to_dataset(
                    book,
                    transformed_file_rootdir,
                    DatasetType::Train,
                    &page.index,
                );
            } else {
                copy_image_to_dataset(
                    book,
                    transformed_file_rootdir,
                    DatasetType::Val,
                    &page.index,
                );
            }

            for text in &page.texts {
                // NOTE: We only care about the rectangle coordinates of the text, not the text (value) itself...
                let (yolo_center_x, yolo_center_y, yolo_width, yolo_height) =
                    convert_boundingbox_to_yolo_xywh(
                        page.width,
                        page.height,
                        text.xmin,
                        text.ymin,
                        text.xmax,
                        text.ymax,
                    );
                write_yolo_label_file(
                    &mut writer,
                    yolo_center_x,
                    yolo_center_y,
                    yolo_width,
                    yolo_height,
                );
            }
        }
        writer.flush().unwrap(); // close?
    }
}
