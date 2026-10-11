from typing import Literal, Optional, Sequence, Tuple

BBoxFormat = Literal[
    "coco",       # [x, y, width, height] in pixels
    "xyxy",       # [xmin, ymin, xmax, ymax] in pixels
    "yolo",       # [x_center, y_center, width, height] normalized
    "cxcywh",     # [x_center, y_center, width, height] in pixels
    "xyxy_norm",  # [xmin, ymin, xmax, ymax] normalized
    "pascal_voc", # [xmin, ymin, xmax, ymax], VOC convention
]

BBox = Tuple[float, float, float, float]

def convert_bbox(
    bbox: Sequence[float],
    from_format: BBoxFormat,
    to_format: BBoxFormat,
    image_size: Optional[Tuple[int, int]] = None,
) -> BBox:
    """
    Convert a bounding box between six common formats.

    Args:
        bbox:
            Four coordinates describing the bounding box.
        from_format:
            Source format: coco, xyxy, yolo, cxcywh,
            xyxy_norm, or pascal_voc.
        to_format:
            Destination format from the same set.
        image_size:
            (image_width, image_height). Required when either
            format is yolo or xyxy_norm.

    Returns:
        A tuple of four coordinates in the destination format.

    Notes:
        - COCO and CXCYWH use pixel coordinates.
        - YOLO and XYXY_NORM use normalized coordinates.
        - Pascal VOC is treated as continuous XYXY here.
        - Coordinates are not clipped to image boundaries.
    """
    formats = {
        "coco",
        "xyxy",
        "yolo",
        "cxcywh",
        "xyxy_norm",
        "pascal_voc",
    }

    if from_format not in formats:
        raise ValueError(f"Unsupported source format: {from_format}")

    if to_format not in formats:
        raise ValueError(f"Unsupported target format: {to_format}")

    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four coordinates.")

    a, b, c, d = map(float, bbox)

    if not all(map(lambda v: v == v and abs(v) != float("inf"),
                   (a, b, c, d))):
        raise ValueError("Bounding box coordinates must be finite.")

    normalized_formats = {"yolo", "xyxy_norm"}

    if (from_format in normalized_formats or to_format in normalized_formats):
        if image_size is None:
            raise ValueError(
                "image_size=(width, height) is required "
                "for normalized formats."
            )

        image_width, image_height = image_size

        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive.")
    else:
        image_width = image_height = None

    # Step 1: Convert source format to pixel-based XYXY.
    if from_format == "coco":
        xmin, ymin = a, b
        xmax, ymax = a + c, b + d

    elif from_format in {"xyxy", "pascal_voc"}:
        xmin, ymin, xmax, ymax = a, b, c, d

    elif from_format == "cxcywh":
        x_center, y_center, width, height = a, b, c, d
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2

    elif from_format == "yolo":
        x_center, y_center, width, height = a, b, c, d

        xmin = (x_center - width / 2) * image_width
        ymin = (y_center - height / 2) * image_height
        xmax = (x_center + width / 2) * image_width
        ymax = (y_center + height / 2) * image_height

    else:  # xyxy_norm
        xmin = a * image_width
        ymin = b * image_height
        xmax = c * image_width
        ymax = d * image_height

    if xmax < xmin or ymax < ymin:
        raise ValueError(
            "Invalid bounding box: xmax must be >= xmin "
            "and ymax must be >= ymin."
        )

    # Step 2: Convert pixel-based XYXY to the target format.
    width = xmax - xmin
    height = ymax - ymin
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    if to_format == "coco":
        result = (xmin, ymin, width, height)

    elif to_format in {"xyxy", "pascal_voc"}:
        result = (xmin, ymin, xmax, ymax)

    elif to_format == "cxcywh":
        result = (x_center, y_center, width, height)

    elif to_format == "yolo":
        result = (
            x_center / image_width,
            y_center / image_height,
            width / image_width,
            height / image_height,
        )

    else:  # xyxy_norm
        result = (
            xmin / image_width,
            ymin / image_height,
            xmax / image_width,
            ymax / image_height,
        )

    return result