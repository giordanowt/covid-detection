import keras
from keras import layers
from keras.applications import VGG19

from covid_detection.manifest import LABELS

DEFAULT_FINE_TUNE_FROM = "block5_conv1"


def build_model(image_size=224, rotation_degrees=20, zoom=0.15, dense_units=64, dropout=0.4, weights="imagenet"):
    base = VGG19(weights=weights, include_top=False, input_shape=(image_size, image_size, 3))
    base.trainable = False
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = layers.RandomRotation(rotation_degrees / 360)(inputs)
    x = layers.RandomZoom(zoom)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(LABELS), activation="softmax")(x)
    return keras.Model(inputs, outputs, name="vgg19_cxr_classifier")


def backbone(model):
    return next(layer for layer in model.layers if isinstance(layer, keras.Model))


def unfreeze_from(model, layer_name=DEFAULT_FINE_TUNE_FROM):
    base = backbone(model)
    base.trainable = True
    trainable = False
    for layer in base.layers:
        if layer.name == layer_name:
            trainable = True
        layer.trainable = trainable
    return base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model
