use anyhow::Result;
use roxmltree;

// annotation_tags = ["frame", "face", "body", "text"]
pub enum AnnotationType {
    Frame,
    Face,
    Body,
    Text,
}

fn normalize_paths(path: &str) -> String {
    path.replace("\\", "/").replace("//", "/")
}

#[derive(Debug)]
pub struct Annotation {
    pub title: String,
    pub characters: Vec<Character>,
    pub pages: Vec<Page>,
}
impl Annotation {
    pub fn new(title: String) -> Self {
        Annotation {
            title,
            characters: Vec::new(),
            pages: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Manga109 {
    pub root_dir: String,
    pub books: Vec<Book>,
}
impl Manga109 {
    //  Manga109 annotation parser
    //  Args:
    //      root_dir (str): The path of the root directory of Manga109 data, e.g., 'YOUR_PATH/Manga109_2017_09_28'
    //  self.root_dir = pathlib.Path(root_dir)
    //  self.books = []  # book titles
    //  with (self.root_dir / "books.txt").open("rt", encoding='utf-8') as f:
    //      self.books = [line.rstrip() for line in f]
    pub fn new(root_dir: &str) -> Self {
        // first, make sure we can find "books.txt" here, it is just a CASE-SENSITIVE (per-line) list
        // of book titles which matches the images sub-directory names as well as annotations XML files
        // For example:
        // root_dir
        // ├── books.txt    // one of the line will be "book1"
        // ├── images
        // │   ├── book1    // dir matches the book title in books.txt
        // │   │   ├── 000.jpg
        // │   │   ├── ...
        // │   ├── ...
        // ├── annotations
        // │   ├── book1.xml    // xml filename matches the book title in books.txt
        // │   ├── ...
        let title_list: Vec<_> = std::fs::read_to_string(format!("{}/books.txt", root_dir))
            .expect("Could not read books.txt")
            .lines()
            .map(|line| line.to_string())
            .collect();
        // if 0 book, panic!
        if title_list.is_empty() {
            panic!("No book titles found in books.txt");
        }
        // Now that books.txt is read, walk through each book titles listed in the file
        let mut books = Vec::new();
        for title in title_list.iter() {
            // For each book title, parse the corresponding XML file
            let book = get_book(root_dir, title).expect("Could not parse XML file");
            books.push(book);
        }

        Manga109 {
            root_dir: root_dir.to_string(),
            books: books, // 1 or more books, if it's empty, we should panic!
        }
    }

    // Given a book title and an index of a page, return the correct image path
    //
    // Args:
    //     book (str): A title of a book. Should be in self.books.
    //     index (int): An index of a page
    //
    // Returns:
    //     str: A path to the selected image
    // assert book in self.books
    // assert isinstance(index, int)
    // return str((self.root_dir / "images" / book / (str(index).zfill(3) + ".jpg")).resolve())  // note: 3 digits jpg file
    pub fn img_path(&self, book: &str, page_index: &usize) -> Result<String> {
        let possible_book = self.books.iter().find(|b| b.title == book);
        // if cannot find, panic
        if possible_book.is_none() {
            panic!("Book not found");
        }
        let book = possible_book.expect("Book '{book}' not found");
        let (index, page) = book
            .pages
            .iter()
            .enumerate()
            .find(|(idx, pg)| idx == page_index)
            .expect("Page {page} of book '{book}' not found");
        let debug_image_root_dir = format!("{}/images/{}", self.root_dir, book.title); // won't use this in the future...
        if debug_image_root_dir != book.get_image_dir_paths() {
            // sometimes, on Windows path, it might have \\ instead of / so it's good to know when/where I've violated the pathing format...
            println!(
                "WARNING: Image root directory mismatch - expected:{}, actual:{}",
                debug_image_root_dir,
                book.get_image_dir_paths()
            );
        }

        // NOTE: jpg files are 3 digits, so we need to zero-pad the index, any pages greater than 999 should have no problems
        // we also assume that exptension ".jpg" are always all lower-case
        let absolute_image_path =
            normalize_paths(format!("{}/{:03}.jpg", book.get_image_dir_paths(), index).as_str());
        // verify if JPG actually exists
        if !std::path::Path::new(&absolute_image_path).exists() {
            panic!("Image '{absolute_image_path}' not found");
        }
        Ok(absolute_image_path)
    }

    //  Given a book title, return its annotations as a dict.
    //
    //  Args:
    //      book (str): The title of the book to get the annotations of.
    //          The title must be contained in the list `self.books`.
    //      annotation_type (str) default `"annotations"` : The directory to load the xml data from.
    //      separate_by_tag (bool) default `True` : When set to `True`, each annotation data type
    //          ("frame", "face", "body", "text") will be stored in a different list in the output
    //          dictionary. When set to `False`, all of the annotation data will be stored in a
    //          single list in the output dictionary. In the latter case, the data in the list will
    //          appear in the same order as in the original XML file.
    //
    //  Returns:
    //      annotation (dict): The annotation data
    pub fn get_annotation(&self, book: &str, separate_by_tag: bool) -> Result<Annotation> {
        let possible_book = self.books.iter().find(|b| b.title == book);
        match possible_book {
            Some(book) => {
                let characters = book.characters.clone();
                let mut pages = Vec::new();
                for page in book.pages.iter() {
                    let mut frames = Vec::new();
                    for frame in page.frames.iter() {
                        frames.push(frame.clone());
                    }

                    let mut texts = Vec::new();
                    for text in page.texts.iter() {
                        texts.push(text.clone());
                    }

                    let mut faces = Vec::new();
                    for face in page.faces.iter() {
                        faces.push(face.clone());
                    }

                    let mut bodies = Vec::new();
                    for body in page.bodies.iter() {
                        bodies.push(body.clone());
                    }

                    pages.push(Page {
                        index: page.index.clone(),
                        width: page.width.clone(),
                        height: page.height.clone(),
                        frames: frames,
                        texts: texts,
                        faces: faces,
                        bodies: bodies,
                    });
                }
                // if there are 0 pages, panic
                if pages.is_empty() {
                    panic!("No pages found in the book");
                }

                Ok(Annotation {
                    title: book.title.clone(),
                    characters,
                    pages,
                })
            }
            None => Err(anyhow::anyhow!("Book not found")),
        }
    }
}

// I'm currently unsure whether to use roxmltree (read-only XML) or quick_xml (read-write XML) for
// deserializing XML in harmony with serde...

// Few notes about the annotation (XML) file:
// * numbers are stored as quoated strings
// * there are no sub-structure called "rectangle" (or "region", whatever) which could have wrapped xmin/ymin/xmax/ymax,
//   but because they are flattened, for deserialization, I cannot structure it as a struct with those fields
// TODO: add deserialization macros documented for quick_xml::de so that we an just deserialize directly
#[derive(Debug)]
pub struct Book {
    pub characters: Vec<Character>,
    pub pages: Vec<Page>,
    pub title: String, // title of the book (case-sensitive)

    // Absolute paths to the annotation XML file and the root directory of images
    // they are usually in format of "{root_dir}/annotations/{book.title}.xml" and "{root_dir}/images"
    // NOTE: title is used as dir-paths, hence it is case sensitive!
    annotation_filepaths: String, // Full paths with filename of the XML file (i.e. /foo/annotations/bar.xml)
    image_root_dir: String,       // note that this is different from img_path()
}
impl Book {
    pub fn get_image_dir_paths(&self) -> String {
        normalize_paths(self.image_root_dir.clone().as_str())
    }
    pub fn get_annotation_file_paths(&self) -> String {
        normalize_paths(self.annotation_filepaths.clone().as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Character {
    pub id: String,
    pub name: String,
}

#[derive(Debug)]
pub struct Page {
    pub frames: Vec<Frame>,
    pub texts: Vec<Text>,
    pub faces: Vec<Face>,
    pub bodies: Vec<Body>,
    pub index: usize, // aka page_number
    pub width: u32,   // aka pixel_width
    pub height: u32,  // aka pixel_height
    //image_file_path: String,  // i.e. "/foo/images/bar/009.jpg" (index==9)
}
impl Clone for Page {
    fn clone(&self) -> Self {
        Page {
            frames: self.frames.clone(),
            texts: self.texts.clone(),
            faces: self.faces.clone(),
            bodies: self.bodies.clone(),
            index: self.index.clone(),
            width: self.width.clone(),
            height: self.height.clone(),
        }
    }
}
impl Page {
    pub fn annotate(&self) -> Vec<Vec<String>> {
        let mut annotations = Vec::new();
        for frame in self.frames.clone() {
            annotations.push(frame.annotate());
        }
        for text in self.texts.clone() {
            annotations.push(text.annotate());
        }
        for face in self.faces.clone() {
            annotations.push(face.annotate());
        }
        for body in self.bodies.clone() {
            annotations.push(body.annotate());
        }
        annotations
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Frame {
    pub id: String, // it's actually an hexadecimal number HASH, but wei'll treate it as a &str
    pub xmin: u32,
    pub ymin: u32,
    pub xmax: u32,
    pub ymax: u32,
}
impl Frame {
    // key-value pairs for the annotation
    // i.e. ["tag", "frame", "id", "0", "xmin", "0", "ymin", "0", "xmax", "0", "ymax", "0"]
    pub fn annotate(&self) -> Vec<String> {
        vec![
            "frame".to_string(),
            self.id.clone(),
            "xmin".to_string(),
            self.xmin.to_string(),
            "ymin".to_string(),
            self.ymin.to_string(),
            "xmax".to_string(),
            self.xmax.to_string(),
            "ymax".to_string(),
            self.ymax.to_string(),
        ]
    }

    pub fn get_width(&self) -> u32 {
        self.xmax - self.xmin
    }
    pub fn get_height(&self) -> u32 {
        self.ymax - self.ymin
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Text {
    pub id: String,
    pub xmin: u32,
    pub ymin: u32,
    pub xmax: u32,
    pub ymax: u32,
    pub value: String,
}
impl Text {
    // key-value pairs for the annotation
    // i.e. ["tag", "text", "id", "0", "xmin", "0", "ymin", "0", "xmax", "0", "ymax", "0", "#text", "value"]
    pub fn annotate(&self) -> Vec<String> {
        vec![
            "text".to_string(),
            self.id.to_string(),
            "xmin".to_string(),
            self.xmin.to_string(),
            "ymin".to_string(),
            self.ymin.to_string(),
            "xmax".to_string(),
            self.xmax.to_string(),
            "ymax".to_string(),
            self.ymax.to_string(),
            "#text".to_string(),
            self.value.clone(),
        ]
    }

    pub fn get_width(&self) -> u32 {
        self.xmax - self.xmin
    }
    pub fn get_height(&self) -> u32 {
        self.ymax - self.ymin
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Face {
    pub id: String,
    pub xmin: u32,
    pub ymin: u32,
    pub xmax: u32,
    pub ymax: u32,
    pub character: String,
}
impl Face {
    // key-value pairs for the annotation
    // i.e. ["tag", "face", "id", "0", "xmin", "0", "ymin", "0", "xmax", "0", "ymax", "0", "character", "character"]
    pub fn annotate(&self) -> Vec<String> {
        vec![
            "face".to_string(),
            self.id.to_string(),
            "xmin".to_string(),
            self.xmin.to_string(),
            "ymin".to_string(),
            self.ymin.to_string(),
            "xmax".to_string(),
            self.xmax.to_string(),
            "ymax".to_string(),
            self.ymax.to_string(),
            "character".to_string(),
            self.character.clone(),
        ]
    }

    pub fn get_width(&self) -> u32 {
        self.xmax - self.xmin
    }
    pub fn get_height(&self) -> u32 {
        self.ymax - self.ymin
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Body {
    pub id: String,
    pub xmin: u32,
    pub ymin: u32,
    pub xmax: u32,
    pub ymax: u32,
    pub character: String,
}
impl Body {
    // key-value pairs for the annotation
    // i.e. ["tag", "body", "id", "0", "xmin", "0", "ymin", "0", "xmax", "0", "ymax", "0", "character", "character"]
    pub fn annotate(&self) -> Vec<String> {
        vec![
            "body".to_string(),
            self.id.clone(),
            "xmin".to_string(),
            self.xmin.to_string(),
            "ymin".to_string(),
            self.ymin.to_string(),
            "xmax".to_string(),
            self.xmax.to_string(),
            "ymax".to_string(),
            self.ymax.to_string(),
            "character".to_string(),
            self.character.clone(),
        ]
    }

    pub fn get_width(&self) -> u32 {
        self.xmax - self.xmin
    }
    pub fn get_height(&self) -> u32 {
        self.ymax - self.ymin
    }
}

pub fn get_book(root_dir: &str, book_title: &str) -> Result<Book, String> {
    println!(
        "Parsing annotation file from '{}/annotations/{}.xml'",
        root_dir, book_title
    );

    let path = std::path::Path::new(root_dir);
    // let's panic if root dir, annotations, or images are not found
    if !path.exists() {
        panic!("Root directory not found");
    }
    let annotations_root_dir = path.join("annotations");
    if !annotations_root_dir.exists() {
        panic!("Annotations directory not found");
    }
    let images_root_dir = path.join("images");
    if !images_root_dir.exists() {
        panic!("Images directory not found");
    }

    // see if BOOK exists
    let annotations_xml_file_paths = annotations_root_dir.join(format!("{}.xml", book_title));
    if !annotations_xml_file_paths.exists() {
        panic!(
            "Annotation XML file '{:?}' not found",
            annotations_xml_file_paths
        );
    }
    let images_for_book_dir = images_root_dir.join(book_title);
    if !images_for_book_dir.exists() {
        panic!(
            "Images directory '{:?}' for book not found",
            images_for_book_dir
        );
    }

    parse_annotation_file_and_make_book(&annotations_xml_file_paths, &images_for_book_dir)
}

fn parse_annotation_file_and_make_book(
    xml_paths: &std::path::Path,
    image_path: &std::path::Path,
) -> Result<Book, String> {
    // panic if the XML file is not found
    if !xml_paths.exists() {
        panic!("XML file '{:?}' not found", xml_paths);
    }
    if !image_path.exists() {
        panic!("Images directory '{:?}' not found", image_path);
    }
    let raw_xml = std::fs::read_to_string(xml_paths).map_err(|e| e.to_string())?;
    parse_raw_xml_annotations_and_make_book(
        &raw_xml,
        &normalize_paths(xml_paths.to_str().unwrap()).as_str(),
        normalize_paths(image_path.to_str().unwrap()).as_str(),
    )
}

// Takes in RAW XML string, makes it easier to unit-test witthout reading from a file directly
fn parse_raw_xml_annotations_and_make_book(
    raw_xml: &str,
    xml_pathsname: &str,
    image_pathsname: &str,
) -> Result<Book, String> {
    // because the format is very trivial, for now, I'll just use roxmltree
    let xml = roxmltree::Document::parse(raw_xml).map_err(|e| e.to_string())?;
    let root = xml.root_element();

    let title = root.attribute("title").ok_or("title not found")?;
    let mut book = Book {
        characters: Vec::new(),
        pages: Vec::new(),
        title: title.to_string(),
        annotation_filepaths: xml_pathsname.to_string(),
        image_root_dir: image_pathsname.to_string(),
    };

    let mut characters = Vec::new();
    for character in root.descendants().filter(|n| n.has_tag_name("character")) {
        let id = character.attribute("id").unwrap().to_string();
        let name = character.attribute("name").unwrap().to_string();
        characters.push(Character { id: id, name: name });
    }
    book.characters = characters;

    let mut pages = Vec::new();
    for page in root.descendants().filter(|n| n.has_tag_name("page")) {
        let index = page
            .attribute("index")
            .unwrap()
            .to_string()
            .parse::<usize>()
            .expect("Pages: Expected usize for index");
        let width = page
            .attribute("width")
            .unwrap()
            .to_string()
            .parse::<u32>()
            .expect("Pages: Expected u32 for width");
        let height = page
            .attribute("height")
            .unwrap()
            .to_string()
            .parse::<u32>()
            .expect("Pages: Expected u32 for height");

        let mut frames = Vec::new();
        for frame in page.descendants().filter(|n| n.has_tag_name("frame")) {
            let id = frame.attribute("id").unwrap().to_string();
            let xmin = frame
                .attribute("xmin")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Frames: Expected u32 for id={id} for xmin");
            let ymin = frame
                .attribute("ymin")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Frames: Expected u32 for id={id} for ymin");
            let xmax = frame
                .attribute("xmax")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Frames: Expected u32 for id={id} for xmax");
            let ymax = frame
                .attribute("ymax")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Frames: Expected u32 for id={id} for ymax");
            frames.push(Frame {
                id: id,
                xmin: xmin,
                ymin: ymin,
                xmax: xmax,
                ymax: ymax,
            });
        }

        let mut texts = Vec::new();
        for text in page.descendants().filter(|n| n.has_tag_name("text")) {
            let id = text.attribute("id").unwrap().to_string();
            let xmin = text
                .attribute("xmin")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Texts: Expected u32 for id={id} for xmin");
            let ymin = text
                .attribute("ymin")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Texts: Expected u32 for id={id} for ymin");
            let xmax = text
                .attribute("xmax")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Texts: Expected u32 for id={id} for xmax");
            let ymax = text
                .attribute("ymax")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Texts: Expected u32 for id={id} for ymax");
            let value = text.text().unwrap().to_string();
            texts.push(Text {
                id: id,
                xmin: xmin,
                ymin: ymin,
                xmax: xmax,
                ymax: ymax,
                value: value,
            });
        }

        let mut faces = Vec::new();
        for face in page.descendants().filter(|n| n.has_tag_name("face")) {
            let id = face.attribute("id").unwrap().to_string();
            let xmin = face
                .attribute("xmin")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Texts: Expected u32 for id={id} for xmin");
            let ymin = face
                .attribute("ymin")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Texts: Expected u32 for id={id} for ymin");
            let xmax = face
                .attribute("xmax")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Texts: Expected u32 for id={id} for xmax");
            let ymax = face
                .attribute("ymax")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Texts: Expected u32 for id={id} for ymax");
            let character = face.attribute("character").unwrap().to_string();
            faces.push(Face {
                id: id,
                xmin: xmin,
                ymin: ymin,
                xmax: xmax,
                ymax: ymax,
                character: character,
            });
        }

        let mut bodies = Vec::new();
        for body in page.descendants().filter(|n| n.has_tag_name("body")) {
            let id = body.attribute("id").unwrap().to_string();
            let xmin = body
                .attribute("xmin")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Bodies: Expected u32 for id={id} for xmin");
            let ymin = body
                .attribute("ymin")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Bodies: Expected u32 for id={id} for ymin");
            let xmax = body
                .attribute("xmax")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Bodies: Expected u32 for id={id} for xmax");
            let ymax = body
                .attribute("ymax")
                .unwrap()
                .to_string()
                .parse::<u32>()
                .expect("Bodies: Expected u32 for id={id} for ymax");
            let character = body.attribute("character").unwrap().to_string();
            bodies.push(Body {
                id: id,
                xmin: xmin,
                ymin: ymin,
                xmax: xmax,
                ymax: ymax,
                character: character,
            });
        }

        pages.push(Page {
            index: index,
            width: width,
            height: height,
            frames: frames,
            texts: texts,
            faces: faces,
            bodies: bodies,
        });
    }
    book.pages = pages;

    Ok(book)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_annotation() {
        let xml = r#"
            <annotation title="title">
                <characters>
                    <character id="id1" name="name1"/>
                    <character id="id2" name="name2"/>
                </characters>
                <pages>
                    <page index="1" width="100" height="200">
                        <frame id="1" xmin="10" ymin="20" xmax="30" ymax="40"/>
                        <text id="1" xmin="10" ymin="20" xmax="30" ymax="40">value1</text>
                        <face id="1" xmin="10" ymin="20" xmax="30" ymax="40" character="character1"/>
                        <body id="1" xmin="10" ymin="20" xmax="30" ymax="40" character="character1"/>
                    </page>
                    <page index="2" width="200" height="300">
                        <frame id="2" xmin="20" ymin="30" xmax="40" ymax="50"/>
                        <text id="2" xmin="20" ymin="30" xmax="40" ymax="50">value2</text>
                        <face id="2" xmin="20" ymin="30" xmax="40" ymax="50" character="character2"/>
                        <body id="2" xmin="20" ymin="30" xmax="40" ymax="50" character="character2"/>
                    </page>
                </pages>
            </annotation>
        "#;

        let book = parse_raw_xml_annotations_and_make_book(xml, "", "").unwrap();
        assert_eq!(book.title, ("title".to_string()));

        let characters = book.characters;
        assert_eq!(characters.len(), 2);
        assert_eq!(characters[0].id, ("id1".to_string()));
        assert_eq!(characters[0].name, ("name1".to_string()));
        assert_eq!(characters[1].id, ("id2".to_string()));
        assert_eq!(characters[1].name, ("name2".to_string()));

        let pages = book.pages;
        assert_eq!(pages.len(), 2);

        let page1 = &pages[0];
        assert_eq!(page1.index, 1);
        assert_eq!(page1.width, 100);
        assert_eq!(page1.height, 200);

        let frames = page1.frames.clone();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].id, "1".to_string());
        assert_eq!(frames[0].xmin, 10);
        assert_eq!(frames[0].ymin, 20);
        assert_eq!(frames[0].xmax, 30);
        assert_eq!(frames[0].ymax, 40);

        let texts = page1.texts.clone();
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0].id, "1".to_string());
        assert_eq!(texts[0].xmin, 10);
        assert_eq!(texts[0].ymin, 20);
        assert_eq!(texts[0].xmax, 30);
        assert_eq!(texts[0].ymax, 40);
        assert_eq!(texts[0].value, "value1".to_string());

        let faces = page1.faces.clone();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].id, "1".to_string());
        assert_eq!(faces[0].xmin, 10);
        assert_eq!(faces[0].ymin, 20);
        assert_eq!(faces[0].xmax, 30);
        assert_eq!(faces[0].ymax, 40);
        assert_eq!(faces[0].character, "character1".to_string());

        let bodies = page1.bodies.clone();
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0].id, "1".to_string());
        assert_eq!(bodies[0].xmin, 10);
        assert_eq!(bodies[0].ymin, 20);
        assert_eq!(bodies[0].xmax, 30);
        assert_eq!(bodies[0].ymax, 40);
        assert_eq!(bodies[0].character, "character1".to_string());

        let page2 = &pages[1];
        assert_eq!(page2.index, 2);
        assert_eq!(page2.width, 200);
        assert_eq!(page2.height, 300);

        let frames = page2.frames.clone();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].id, "2".to_string());
        assert_eq!(frames[0].xmin, 20);
        assert_eq!(frames[0].ymin, 30);

        let texts = page2.texts.clone();
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0].id, "2".to_string());
        assert_eq!(texts[0].xmin, 20);
        assert_eq!(texts[0].ymin, 30);
        assert_eq!(texts[0].xmax, 40);
        assert_eq!(texts[0].ymax, 50);
        assert_eq!(texts[0].value, "value2".to_string());

        let faces = page2.faces.clone();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].id, "2".to_string());
        assert_eq!(faces[0].xmin, 20);
        assert_eq!(faces[0].ymin, 30);
        assert_eq!(faces[0].xmax, 40);
        assert_eq!(faces[0].ymax, 50);
        assert_eq!(faces[0].character, "character2".to_string());

        let bodies = page2.bodies.clone();
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0].id, "2".to_string());
        assert_eq!(bodies[0].xmin, 20);
        assert_eq!(bodies[0].ymin, 30);
        assert_eq!(bodies[0].xmax, 40);
        assert_eq!(bodies[0].ymax, 50);
        assert_eq!(bodies[0].character, "character2".to_string());
    }
}
