"""Minimal reusable-API example for the 12-class defect classifier."""

from model_classify.predict import DefectClassifier, read_image_bgr


classifier = DefectClassifier(
    backbone_path=(
        "model_matcher/onnx_dino_v3_b16_finegrained_multiscale_equal50_fn1_v6_clean/"
        "backbone.onnx"
    ),
    head_path=(
        "model_classify/checkpoints_dino_v3_b16_classify_v1/"
        "classifier_head.onnx"
    ),
    config_path=(
        "model_classify/checkpoints_dino_v3_b16_classify_v1/model_config.json"
    ),
    device="cpu",
    batch_size=3,
)

images = [
    read_image_bgr("image1.jpg"),
    read_image_bgr("image2.jpg"),
    read_image_bgr("image3.jpg"),
]

if any(image is None for image in images):
    raise FileNotFoundError("Có ảnh không đọc được")

results = classifier.predict(images)
for result in results:
    print(result.class_id, result.class_name, result.confidence)
