from __future__ import annotations

import numpy as np
import pytest

from ml_pipes.vision import BlendImages, ConvertColorSpace, ImagePayload, Normalize, Resize


def test_resize_op_can_do_plain_resize_without_padding():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")

    resized, transform = Resize(target_size=(40, 40), mode="resize")(payload)

    assert resized.array.shape == (40, 40, 3)
    assert transform.scale == (2.0, 4.0)
    assert transform.pad == (0.0, 0.0)
    assert transform.resized_shape == (40, 40)


def test_image_payload_exposes_derived_shape_properties():
    payload = ImagePayload(array=np.zeros((10, 20, 3), dtype=np.uint8), color_space="BGR", layout="HWC")

    assert payload.shape == (10, 20, 3)
    assert payload.spatial_shape == (10, 20)
    assert payload.height == 10
    assert payload.width == 20
    assert payload.size == (20, 10)
    assert payload.dtype == "uint8"
    assert payload.ndim == 3
    assert payload.channels == 3


def test_image_payload_spatial_shape_uses_layout_for_chw():
    payload = ImagePayload(array=np.zeros((3, 10, 20), dtype=np.uint8), color_space="BGR", layout="CHW")

    assert payload.shape == (3, 10, 20)
    assert payload.spatial_shape == (10, 20)
    assert payload.height == 10
    assert payload.width == 20
    assert payload.size == (20, 10)
    assert payload.channels == 3


def test_convert_color_space_converts_bgr_to_rgb_and_preserves_metadata():
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")

    converted = ConvertColorSpace("RGB")(payload)

    assert converted.color_space == "RGB"
    assert converted.layout == "HWC"
    assert converted.dtype == "uint8"
    assert converted.array.flags.c_contiguous
    assert converted.array.tolist() == [[[30, 20, 10], [60, 50, 40]]]


def test_convert_color_space_uses_channel_axis_from_layout():
    image = np.array(
        [
            [[10, 40]],
            [[20, 50]],
            [[30, 60]],
        ],
        dtype=np.uint8,
    )
    payload = ImagePayload(array=image, color_space="BGR", layout="CHW")

    converted = ConvertColorSpace("RGB")(payload)

    assert converted.layout == "CHW"
    assert converted.color_space == "RGB"
    assert converted.array.flags.c_contiguous
    assert converted.array.tolist() == [[[30, 60]], [[20, 50]], [[10, 40]]]


def test_convert_color_space_rejects_non_three_channel_input():
    payload = ImagePayload(array=np.zeros((10, 20, 1), dtype=np.uint8), color_space="BGR", layout="HWC")

    with pytest.raises(ValueError, match="3-channel"):
        ConvertColorSpace("RGB")(payload)


def test_convert_color_space_rejects_unknown_source_color_space():
    payload = ImagePayload(array=np.zeros((10, 20, 3), dtype=np.uint8), color_space="HSV", layout="HWC")

    with pytest.raises(ValueError, match="BGR/RGB input"):
        ConvertColorSpace("RGB")(payload)


def test_blend_images_preserves_shape_dtype_and_metadata() -> None:
    source = ImagePayload(array=np.zeros((4, 5, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    overlay = ImagePayload(array=np.full((4, 5, 3), 255, dtype=np.uint8), color_space="BGR", layout="HWC")

    result = BlendImages()(source, overlay)

    assert result.array.shape == source.array.shape
    assert result.array.dtype == np.uint8
    assert result.color_space == source.color_space


def test_normalize_op_can_keep_bgr_and_hwc_without_batch():
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")

    tensor = Normalize(
        output_color_space="BGR",
        output_layout="HWC",
        add_batch_dim=False,
        scale=1.0,
    )(payload)

    assert tensor.layout == "HWC"
    assert tensor.dtype == "float32"
    assert tensor.array.shape == (1, 1, 3)
    assert tensor.array.tolist() == [[[10.0, 20.0, 30.0]]]


def test_normalize_op_preserves_floating_input_dtype():
    image = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float16)
    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")

    tensor = Normalize(
        output_color_space="BGR",
        output_layout="HWC",
        add_batch_dim=False,
        scale=1.0,
    )(payload)

    assert tensor.array.dtype == np.float16
    assert tensor.dtype == "float16"
