import sys, os
from typing import Optional, TYPE_CHECKING
from ultralytics.engine.results import Results, Boxes
import logging
import numpy as np
import torch
from copy import deepcopy
from ultralytics.utils.plotting import Annotator, colors

if TYPE_CHECKING:
    from consolidator import Consolidator

def get_logger(name: str, log_path: str):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        print("Logger already has handlers", file=sys.stderr)
        return logger
    
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        f"%(asctime)s - %(name)s::%(levelname)s %(message)s",
        "%H:%M:%S"
    )

    os.makedirs(log_path, exist_ok=True)
    
    file_handler = logging.FileHandler(f"{log_path}/{name}.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def crop_box_from_result(result: Results, box: Boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    crop = result.orig_img[y1:y2, x1:x2].copy()
    return crop

def get_classification_class(results: list[Results]):
    result = results[0]
    class_id = result.probs.top1
    class_name = result.names[class_id]
    return class_id, class_name

def get_detection_class(result: Results, box: Boxes):
    class_id = int(box.cls.item())
    class_name = result.names[class_id]
    return class_id, class_name

def get_unique_path(save_path: str):
    i = 2
    _save_path = save_path
    while os.path.exists(_save_path):
        _save_path = f"{save_path}_{i}"
        i += 1
    os.makedirs(_save_path, exist_ok=True)
    return _save_path

def plot_from_track_results(
    result: Results,
    consolidator: "Optional[Consolidator]",
    conf: bool = True,
    line_width: Optional[int] = None,
    font_size: Optional[int] = None,
    font: str = "Arial.ttf",
    color_mode: str = "class",
) -> np.ndarray:
    """
    Plot detection results on an input RGB image. A simplified version of result.plot()

    Args:
        conf (bool): Whether to plot detection confidence scores.
        line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
        font_size (float | None): Font size for text. If None, scaled to image size.
        font (str): Font to use for text.
        color_mode (str): Specify the color mode, e.g., 'instance' or 'class'.

    Returns:
        (np.ndarray): Annotated image as a numpy array.

    Examples:
        >>> results = model("image.jpg")
        >>> for result in results:
        >>>     im = result.plot()
        >>>     im.show()
    """
    assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
    if isinstance(result.orig_img, torch.Tensor):
        img = (result.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()
    else:
        img = result.orig_img

    pred_boxes = result.boxes
    annotator = Annotator(
        deepcopy(img),
        line_width,
        font_size,
        font,
    )

    # Plot Detect results
    if pred_boxes is not None:
        for d in reversed(pred_boxes):
            if not d.id: continue
            original_id = int(d.id.item())
            class_id, class_name = get_detection_class(result, d)
            id = consolidator.get_representative_id(class_name, original_id) if consolidator else original_id
            name = f"{class_name}#{id}"
            if id != original_id: name += f"({original_id})"
            label = f"{name} {int(d.conf*100)}%" if conf else name
            box = d.xyxy.squeeze()
            annotator.box_label(
                box, label,
                color=colors(class_id, True),
            )

    return annotator.result()
