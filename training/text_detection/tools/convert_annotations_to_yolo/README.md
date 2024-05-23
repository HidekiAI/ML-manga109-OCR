# Data transformations

From what I understand, there are 3 formats to bounding boxes:

```python
# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(YOLO format)
# `ltwh` means left top and width, height(COCO format)
_formats = ["xyxy", "xywh", "ltwh"]
```

This is why you see tools/[utils](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py) in YOLO refering to `xyxy` or `xywh`...  there is also `xywhr` (I think that's Tensor?) which adds rotation to the mix (I think it's clever, but I don't know whether r is radian or rotation (in degrees) - note that utils has calculations of `angle` to be `angle / 180 * PI` so I think it's degrees instead of radian)...

Manga109 annotations uses absolute pixel coordinates, while the YOLO format (2nd) is based on normalized coordinates of range `[0.0..1.0]`, as well as width and height to be normalized (or is it?!?)  Logically, it makes sense to normalize width and height as well (i.e. if original image's width was 640, and bounding box width is image_width=320, then box_width=0.5, this way if it gets scaled to 1024, then bounding box will be 512).  Incidentally, caveat of using normalized value is that you have to remember whether to normalize based on *original image dimension* or current dimension.  It's not really an issue unless you have to deal with scaling more than once (i.e. you scale down from 1024 down 50% (0.5) and then call scale again to 50%, in first case, 2nd call will refer to 50% of original dimension anyways, so consecutive calls will still remain at 512 pixels; as compared to chaining the scales will cause 1024 -> 512 -> 256 pixels).

Looking at the utils [ops.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py) logic:

```python
def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y
```

I don't know who (the smart people at Tensor, Ultralytics/YOLO, ONNX, or what) came up with using normalized value between `[0.0 .. 1.0]` but IMHO, that's brilliant, well actually, for most of us programmers, that's what makes more sense than to actually use absolute coordinates, mainly because we're used to 3D-Texture mapping or something in that neighbor of dealing with the fact that when we deal with images, we have to normalize it so that we can scale the images and not be concerned on scaling the pixel coordinates each time...

In any case, the above logic to me seems that it's *NOT* normalizing it, it's just calculating midpoints in absolute coordinates (I don't know whether Python will dynamically convert to float on dividing by (I assume) integer, or it truncates with implicit type inferences that it's dividing by int, etc - again, this is why I like Rust, it's not ambigous in this way).  

For now, my conventions are as follows:

```rust
// Coordinates are absolute image coordinates found in Manga109 annotations
// Manga109 deals with rectangles that are assumes that origin of the image
// is at upper-left corner of the image, making all rectangles have positive
// coordinates.
// The YOLO xywh format is normalized to range [0.0 .. 1.0] (note it's positive only)
fn convert_boundingbox_to_yolo_xywh(
    image_width: u32,   // we don't care whether this dimension is of original image or resized image!
    image_height: u32,
    rect_minx: u32,
    rect_miny: u32,
    rect_maxx: u32,
    rect_maxy: u32,
) -> (f32/*center_x*/, f32/*center_y*/, f32/*width*/, f32/*height*/) {
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
    (yolo_center_x, yolo_center_y, yolo_width, yolo_height)
}
```

Note that above logic is not validating the input value as well as output (i.e. it's a bug from caller if they set width=0), the logic in `main.rs` will `panic!()` on bad values.

I'm using my own version of [Manga109API in rust](../../../../Prototypes/manga109api/README.md) which basically is a close-but-not-quite port of the Python version...

Because I've no clue whether YOLO has similar effect as TensorFlow in which subdirectory-names is the classification (i.e. `images\apples`, `images\oranges`), I'll flatten the directory and see how many jpg files I can stuff in single directory...  Directory structure will be as following:

```text
data/
├── images/
│   ├── train/
│   │   ├── <title_t1>_001.jpg
│   │   ├── <title_t1>_002.jpg
│   │   ├── ...
│   │   └── <title_tN>_<page>.jpg
│   └── val/
│       ├── <title_v1>_001.jpg
│       ├── <title_v1>_002.jpg
│       ├── ...
│       └── <title_vN>_<page>.jpg
└── labels/
    ├── train/
    │   ├── <title_t1>.txt
    │   ├── ...
    │   └── <title_tN>.txt
    └── val/
        ├── <title_v1>.txt
        ├── ...
        └── <title_vN>.txt
```

labels are in YOLO annotation text format of:

```text
    <category> <center_x> <center_y> <width> <height>
```

where `<center_x> <center_y> <width> <height>` are normalized to `[0.0 .. 1.0]` and `<category>` is the category index of the object;  In this case, since there will be only one category ("TEXT"), it will (always) be 0.

And finally, this part is mainly to train to detect the text, so the `value` element (that actual Japanese UTF-8 string inside the bounding box) of the `text` annotations are ignored.
