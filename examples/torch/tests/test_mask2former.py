from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "packages" / "core" / "src").is_dir()
        ):
            return candidate
    raise RuntimeError("Could not locate repository root")


PROJECT_ROOT = _repo_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.torch.mask2former_infer import (  # noqa: E402
    Mask2FormerBundle,
    PrepareHFImageInputs,
    build_mask2former_infer_pipeline,
)
from examples.torch.run_mask2former_numpy_postprocess import PanopticSegmentsFromQueries  # noqa: E402
from examples.torch.run_mask2former_torch_postprocess import TorchPanopticSegmentsFromQueries  # noqa: E402
from ml_pipes.core import Pipeline  # noqa: E402
from ml_pipes.standard import Recall  # noqa: E402
from ml_pipes.tensor import ArgMax  # noqa: E402
from ml_pipes.tensor import TensorPayload, TensorRegistry  # noqa: E402
from ml_pipes.torch import ToTorch, TorchArgMax, TorchExtract, TorchInfer, TorchSqueeze  # noqa: E402
from ml_pipes.torch.types import TorchTensorRegistry  # noqa: E402
from ml_pipes.vision import ImagePayload  # noqa: E402


def _write_png(path: Path, image: np.ndarray) -> None:
    cv2 = pytest.importorskip("cv2")
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


class _FakeProcessor:
    def __init__(self) -> None:
        self.last_image: np.ndarray | None = None
        self.last_return_tensors: str | None = None

    def __call__(self, *, images: np.ndarray, return_tensors: str) -> dict[str, np.ndarray]:
        self.last_image = images
        self.last_return_tensors = return_tensors
        pixel_values = np.transpose(images, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
        return {"pixel_values": pixel_values}


class _FakeModel(torch.nn.Module):
    def forward(self, *, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = pixel_values.shape[0]
        device = pixel_values.device
        return {
            "class_queries_logits": torch.arange(batch * 6, dtype=torch.float32, device=device).reshape(batch, 2, 3),
            "masks_queries_logits": torch.arange(batch * 24, dtype=torch.float32, device=device).reshape(batch, 2, 3, 4),
        }


def test_mask2former_infer_pipeline_preserves_stored_values_and_exposes_model_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    _write_png(image_path, image)
    processor = _FakeProcessor()
    bundle = Mask2FormerBundle(
        task="instance",
        model_id="fake/mask2former",
        processor=processor,
        model=_FakeModel(),
        class_names=["a", "b"],
        thing_class_ids=frozenset({0, 1}),
    )

    pipeline = build_mask2former_infer_pipeline(bundle, "cpu") + Pipeline([Recall("source_image"), Recall("image_shape")])

    registry, source_image, image_shape = pipeline(image_path)

    assert processor.last_image is not None
    assert processor.last_return_tensors == "np"
    assert processor.last_image.tolist() == [[[30, 20, 10], [60, 50, 40]]]
    assert isinstance(registry, TorchTensorRegistry)
    assert tuple(registry["class_queries_logits"].shape) == (2, 3)
    assert tuple(registry["masks_queries_logits"].shape) == (2, 3, 4)
    assert source_image.color_space == "BGR"
    assert source_image.layout == "HWC"
    assert source_image.array.tolist() == image.tolist()
    assert image_shape == (1, 2)


def test_prepare_hf_image_inputs_consumes_hugging_face_ready_array() -> None:
    processor = _FakeProcessor()
    image = ImagePayload(array=np.zeros((3, 4, 3), dtype=np.uint8), color_space="RGB", layout="HWC")

    result = PrepareHFImageInputs(processor=processor, output_key="pixel_values")(image)

    assert processor.last_image is image.array
    assert processor.last_return_tensors == "np"
    assert isinstance(result, TensorPayload)
    assert result.layout == "NCHW"
    assert result.dtype == "float32"
    assert tuple(result.array.shape) == (1, 3, 3, 4)


def test_prepare_hf_image_inputs_rejects_non_contiguous_array() -> None:
    processor = _FakeProcessor()
    image = ImagePayload(
        array=np.zeros((4, 4, 3), dtype=np.uint8)[:, ::2, :],
        color_space="RGB",
        layout="HWC",
    )

    assert not image.array.flags.c_contiguous

    with pytest.raises(ValueError, match="contiguous image array"):
        PrepareHFImageInputs(processor=processor, output_key="pixel_values")(image)


def test_prepare_hf_image_inputs_supports_custom_contract() -> None:
    class _CustomProcessor:
        def __call__(self, *, images: np.ndarray, return_tensors: str) -> dict[str, np.ndarray]:
            assert return_tensors == "np"
            return {"inputs": images[None, ...].astype(np.float32, copy=False)}

    image = ImagePayload(
        array=np.zeros((3, 5, 7), dtype=np.uint8),
        color_space="RGB",
        layout="CHW",
    )

    result = PrepareHFImageInputs(
        processor=_CustomProcessor(),
        output_key="inputs",
        input_layout="CHW",
        input_channels=3,
        output_layout="NCHW",
        require_contiguous=False,
    )(image)

    assert isinstance(result, TensorPayload)
    assert result.layout == "NCHW"
    assert result.dtype == "float32"
    assert tuple(result.array.shape) == (1, 3, 5, 7)


def test_mask2former_torch_infer_exposes_model_outputs() -> None:
    processor = _FakeProcessor()
    image = ImagePayload(array=np.zeros((3, 4, 3), dtype=np.uint8), color_space="RGB", layout="HWC")

    pixel_values = PrepareHFImageInputs(processor=processor, output_key="pixel_values")(image)
    outputs = TorchInfer(
        _FakeModel(),
        input_name="pixel_values",
        input_layout="NCHW",
    )(ToTorch(device="cpu")(pixel_values))
    registry = TorchExtract("class_queries_logits", "masks_queries_logits")(outputs)
    TorchSqueeze("class_queries_logits", axis=0)(registry)
    TorchSqueeze("masks_queries_logits", axis=0)(registry)

    assert tuple(registry["class_queries_logits"].shape) == (2, 3)
    assert tuple(registry["masks_queries_logits"].shape) == (2, 3, 4)


def test_mask2former_boundary_pipeline_validates() -> None:
    processor = _FakeProcessor()
    bundle = Mask2FormerBundle(
        task="instance",
        model_id="fake/mask2former",
        processor=processor,
        model=_FakeModel(),
        class_names=["a", "b"],
        thing_class_ids=frozenset({0, 1}),
    )

    contract = (
        build_mask2former_infer_pipeline(bundle, "cpu")
    ).validate()

    assert contract is not None


def test_numpy_panoptic_segments_filter_by_surviving_overlap() -> None:
    registry = TensorRegistry(
        {
            "query_scores": np.array([0.9, 0.8], dtype=np.float32),
            "query_classes": np.array([0, 1], dtype=np.int64),
            "mask_probs": np.array(
                [
                    [
                        [0.9, 0.9, 0.0],
                        [0.9, 0.9, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.9, 0.9],
                        [0.0, 0.9, 0.9],
                    ],
                ],
                dtype=np.float32,
            ),
            "winner_ids": np.array(
                [
                    [0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1],
                ],
                dtype=np.int64,
            ),
        }
    )

    result = PanopticSegmentsFromQueries(
        thing_class_ids=frozenset({0, 1}),
        scores="query_scores",
        classes="query_classes",
        masks="mask_probs",
        winner_ids="winner_ids",
        overlap_threshold=0.8,
    )(registry)

    np.testing.assert_array_equal(result["classes"], np.array([1], dtype=np.int64))
    np.testing.assert_allclose(result["scores"], np.array([0.8], dtype=np.float32))
    np.testing.assert_array_equal(
        result["masks"],
        np.array(
            [
                [
                    [False, False, False],
                    [False, True, True],
                    [False, True, True],
                ]
            ],
            dtype=bool,
        ),
    )


def test_torch_panoptic_segments_filter_by_surviving_overlap() -> None:
    registry = TorchTensorRegistry(
        {
            "query_scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
            "query_classes": torch.tensor([0, 1], dtype=torch.int64),
            "mask_probs": torch.tensor(
                [
                    [
                        [0.9, 0.9, 0.0],
                        [0.9, 0.9, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.9, 0.9],
                        [0.0, 0.9, 0.9],
                    ],
                ],
                dtype=torch.float32,
            ),
            "winner_ids": torch.tensor(
                [
                    [0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1],
                ],
                dtype=torch.int64,
            ),
        }
    )

    result = TorchPanopticSegmentsFromQueries(
        thing_class_ids=frozenset({0, 1}),
        scores="query_scores",
        classes="query_classes",
        masks="mask_probs",
        winner_ids="winner_ids",
        overlap_threshold=0.8,
    )(registry)

    assert result["classes"].tolist() == [1]
    assert result["scores"].tolist() == pytest.approx([0.8])
    assert result["masks"].tolist() == [
        [
            [False, False, False],
            [False, True, True],
            [False, True, True],
        ]
    ]


def test_numpy_panoptic_sequence_handles_all_queries_filtered() -> None:
    registry = TensorRegistry(
        {
            "query_scores": np.zeros((0,), dtype=np.float32),
            "query_classes": np.zeros((0,), dtype=np.int64),
            "mask_probs": np.zeros((0, 2, 3), dtype=np.float32),
            "weighted_masks": np.zeros((0, 2, 3), dtype=np.float32),
        }
    )

    ArgMax("weighted_masks", axis=0, as_="winner_ids")(registry)
    result = PanopticSegmentsFromQueries(
        thing_class_ids=frozenset({0, 1}),
        scores="query_scores",
        classes="query_classes",
        masks="mask_probs",
        winner_ids="winner_ids",
    )(registry)

    assert result["winner_ids"].shape == (2, 3)
    assert result["masks"].shape == (0, 2, 3)
    assert result["scores"].dtype == np.float32
    assert result["classes"].dtype == np.int64


def test_torch_panoptic_sequence_handles_all_queries_filtered() -> None:
    registry = TorchTensorRegistry(
        {
            "query_scores": torch.zeros((0,), dtype=torch.float32),
            "query_classes": torch.zeros((0,), dtype=torch.int64),
            "mask_probs": torch.zeros((0, 2, 3), dtype=torch.float32),
            "weighted_masks": torch.zeros((0, 2, 3), dtype=torch.float32),
        }
    )

    TorchArgMax("weighted_masks", axis=0, as_="winner_ids")(registry)
    result = TorchPanopticSegmentsFromQueries(
        thing_class_ids=frozenset({0, 1}),
        scores="query_scores",
        classes="query_classes",
        masks="mask_probs",
        winner_ids="winner_ids",
    )(registry)

    assert tuple(result["winner_ids"].shape) == (2, 3)
    assert tuple(result["masks"].shape) == (0, 2, 3)
    assert result["scores"].dtype == torch.float32
    assert result["classes"].dtype == torch.int64
