"""
vision_utils.py
----------------
Computer-vision disease classification for uploaded crop/leaf images,
using a pretrained Hugging Face image-classification model.

Loads the image processor and model explicitly (rather than via the
generic `pipeline()` auto-detection) because this model repo's config
doesn't expose the image_processor_type field that auto-detection needs.
"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

_processor = None
_model = None

MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"


def get_model_and_processor():
    """Return cached (processor, model) tuple, loaded explicitly."""
    global _processor, _model
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _processor, _model


def classify_leaf_image(image_file, top_k: int = 3) -> list[dict]:
    """
    Run disease classification on an uploaded image.
    Returns a list of {"label": str, "score": float}, sorted by confidence.
    """
    processor, model = get_model_and_processor()

    image = Image.open(image_file).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.shape[0]))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        label = model.config.id2label[idx.item()]
        results.append({"label": label, "score": prob.item()})

    return results