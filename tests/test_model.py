import numpy as np

from covid_detection.model import backbone, build_model, compile_model, unfreeze_from


def test_head_only_model_has_frozen_backbone():
    model = build_model(image_size=64, weights=None)
    assert model.output_shape == (None, 3)
    assert backbone(model).trainable is False


def test_unfreeze_from_block5_keeps_earlier_blocks_frozen():
    model = build_model(image_size=64, weights=None)
    base = unfreeze_from(model, "block5_conv1")
    by_name = {layer.name: layer.trainable for layer in base.layers}
    assert by_name["block4_conv4"] is False
    assert by_name["block5_conv1"] is True
    assert by_name["block5_conv4"] is True


def test_compiled_model_trains_one_step_with_categorical_loss():
    model = compile_model(build_model(image_size=64, weights=None), 1e-3)
    assert model.loss == "categorical_crossentropy"
    x = np.random.rand(4, 64, 64, 3).astype("float32")
    y = np.eye(3)[[0, 1, 2, 0]]
    loss, accuracy = model.train_on_batch(x, y)
    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
