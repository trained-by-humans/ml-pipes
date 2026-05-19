from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.torch.mask2former_infer import (
    LoadedMask2Former,
    Mask2FormerInfer,
    build_mask2former_preprocess_pipeline,
)
from examples.torch.run_mask2former_numpy_postprocess import PanopticSegmentsFromQueries
from examples.torch.run_mask2former_torch_postprocess import TorchPanopticSegmentsFromQueries
from ml_pipes import ArgMax, Pipeline, Recall
from ml_pipes.torch import TorchArgMax
from ml_pipes.torch.types import TorchTensorRegistry
from ml_pipes.types import TensorRegistry


def _write_png(path: Path, image: np.ndarray) -> None:
    cv2 = pytest.importorskip("cv2")
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


class _FakeProcessor:
    def __init__(self) -> None:
        self.last_image: np.ndarray | None = None
        self.last_return_tensors: str | None = None

    def __call__(self, *, images: np.ndarray, return_tensors: str) -> dict[str, torch.Tensor]:
        self.last_image = images
        self.last_return_tensors = return_tensors
        pixel_values = torch.from_numpy(np.transpose(images, (2, 0, 1))).unsqueeze(0).to(dtype=torch.float32)
        return {"pixel_values": pixel_values}


class _FakeModel(torch.nn.Module):
    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        batch = pixel_values.shape[0]
        device = pixel_values.device
        return SimpleNamespace(
            class_queries_logits=torch.arange(batch * 6, dtype=torch.float32, device=device).reshape(batch, 2, 3),
            masks_queries_logits=torch.arange(batch * 24, dtype=torch.float32, device=device).reshape(batch, 2, 3, 4),
        )


def test_mask2former_preprocess_pipeline_produces_rgb_array_and_preserves_stored_values(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    _write_png(image_path, image)

    pipeline = build_mask2former_preprocess_pipeline() + Pipeline([Recall("source_image"), Recall("image_shape")])

    rgb, source_image, image_shape = pipeline(image_path)

    assert isinstance(rgb, np.ndarray)
    assert rgb.flags.c_contiguous
    assert rgb.tolist() == [[[30, 20, 10], [60, 50, 40]]]
    assert source_image.color_space == "BGR"
    assert source_image.layout == "HWC"
    assert source_image.array.tolist() == image.tolist()
    assert image_shape == (1, 2)


def test_mask2former_infer_consumes_hugging_face_ready_array() -> None:
    processor = _FakeProcessor()
    bundle = LoadedMask2Former(
        task="instance",
        model_id="fake/mask2former",
        processor=processor,
        model=_FakeModel(),
        class_names=["a", "b"],
        thing_class_ids=frozenset({0, 1}),
    )
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    result = Mask2FormerInfer(bundle=bundle, device="cpu")(image)

    assert processor.last_image is image
    assert processor.last_return_tensors == "pt"
    assert tuple(result["class_queries_logits"].shape) == (2, 3)
    assert tuple(result["masks_queries_logits"].shape) == (2, 3, 4)


def test_mask2former_infer_rejects_non_contiguous_array() -> None:
    processor = _FakeProcessor()
    bundle = LoadedMask2Former(
        task="instance",
        model_id="fake/mask2former",
        processor=processor,
        model=_FakeModel(),
        class_names=["a", "b"],
        thing_class_ids=frozenset({0, 1}),
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)[:, ::2, :]

    assert not image.flags.c_contiguous

    with pytest.raises(ValueError, match="contiguous RGB ndarray"):
        Mask2FormerInfer(bundle=bundle, device="cpu")(image)


def test_mask2former_boundary_pipeline_validates() -> None:
    processor = _FakeProcessor()
    bundle = LoadedMask2Former(
        task="instance",
        model_id="fake/mask2former",
        processor=processor,
        model=_FakeModel(),
        class_names=["a", "b"],
        thing_class_ids=frozenset({0, 1}),
    )

    contract = (
        build_mask2former_preprocess_pipeline()
        + Pipeline([Mask2FormerInfer(bundle=bundle, device="cpu")])
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
