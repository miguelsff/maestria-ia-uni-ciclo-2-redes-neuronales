"""Mask, weighting, spatial integrity and geography regression tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from neural_exam.flood.data import SPLITS, FloodDataset, build_records, prepare, read_pair
from neural_exam.flood.loss import segmentation_loss
from neural_exam.flood.model import UNet
from neural_exam.flood.pipeline import evaluate


@pytest.mark.parametrize("kind", ["bce", "dice", "combined"])
def test_nodata_loss_has_no_gradient(kind):
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    target = torch.tensor([[[[1.0, -1.0], [0.0, -1.0]]]])
    segmentation_loss(logits, target, torch.ones_like(target), kind).backward()
    assert torch.equal(logits.grad[target == -1], torch.zeros(2))


@pytest.mark.parametrize("kind", ["bce", "dice", "combined"])
def test_weak_weight_survives_weak_only_batches(kind):
    logits, target = torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2)
    strong = segmentation_loss(logits, target, torch.ones_like(target), kind)
    weak = segmentation_loss(logits, target, torch.full_like(target, 0.3), kind)
    assert weak.item() == pytest.approx(strong.item() * 0.3)


def test_all_nodata_batch_is_finite_and_zero():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    loss = segmentation_loss(logits, torch.full_like(logits, -1), torch.ones_like(logits))
    loss.backward()
    assert loss.item() == 0
    assert logits.grad.abs().sum() == 0


def test_unet_preserves_odd_spatial_dimensions():
    model = UNet(base_channels=8).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 2, 33, 35)).shape == (1, 1, 33, 35)


def test_raster_alignment_and_nodata(tmp_path):
    image, label = tmp_path / "sar.tif", tmp_path / "label.tif"
    profile = dict(
        driver="GTiff",
        width=2,
        height=2,
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        dtype="float32",
        nodata=-9999,
    )
    with rasterio.open(image, "w", count=2, **profile) as raster:
        pixels = np.ones((2, 2, 2), dtype=np.float32)
        pixels[0, 0, 0] = -9999
        raster.write(pixels)
    with rasterio.open(label, "w", count=1, **profile) as raster:
        raster.write(np.ones((2, 2), dtype=np.float32), 1)
    record = {"id": "test", "image": str(image), "label": str(label)}
    assert read_pair(record)[1][0, 0] == -1
    with rasterio.open(label, "r+") as raster:
        raster.transform = from_origin(0, 2 + 1e-14, 1 + 1e-15, 1)
    assert read_pair(record)[1][0, 0] == -1
    with rasterio.open(label, "r+") as raster:
        raster.transform = from_origin(0, 3, 1, 1)
    with pytest.raises(ValueError, match="aligned"):
        read_pair(record)


def test_bolivia_in_training_is_rejected(tmp_path):
    split_root = tmp_path / "v1.1/splits/flood_handlabeled"
    split_root.mkdir(parents=True)
    for filename in SPLITS.values():
        (split_root / filename).write_text("", encoding="utf-8")
    (split_root / SPLITS["train"]).write_text("Bolivia_1_S1Hand.tif,Bolivia_1_LabelHand.tif\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Bolivia"):
        build_records(tmp_path, 4)


def _write_pair(root, identifier, value, weak=False, target=None):
    folder = "WeaklyLabeled" if weak else "HandLabeled"
    image_layer, label_layer = ("S1Weak", "S1OtsuLabelWeak") if weak else ("S1Hand", "LabelHand")
    image = root / "v1.1/data/flood_events" / folder / image_layer / f"{identifier}_{image_layer}.tif"
    label = root / "v1.1/data/flood_events" / folder / label_layer / f"{identifier}_{label_layer}.tif"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=2,
        height=2,
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        dtype="float32",
        nodata=-9999,
    )
    with rasterio.open(image, "w", count=2, **profile) as raster:
        raster.write(np.full((2, 2, 2), value, dtype=np.float32))
    with rasterio.open(label, "w", count=1, **profile) as raster:
        raster.write(np.ones((2, 2), dtype=np.float32) if target is None else target.astype(np.float32), 1)
    return {
        "id": identifier,
        "image": str(image),
        "label": str(label),
        "weight": 1.0,
        "country": identifier.split("_")[0],
    }


def _official_fixture(root):
    split_root = root / "v1.1/splits/flood_handlabeled"
    split_root.mkdir(parents=True)
    for split, identifier in zip(SPLITS, ["Ghana_1", "India_2", "Spain_3", "Bolivia_4"], strict=True):
        _write_pair(root, identifier, -10 if split == "train" else 10)
        (split_root / SPLITS[split]).write_text(
            f"{identifier}_S1Hand.tif,{identifier}_LabelHand.tif\n", encoding="utf-8"
        )


def test_normalization_uses_gold_training_only(tmp_path):
    _official_fixture(tmp_path)
    _write_pair(tmp_path, "Ghana_5", 10, weak=True)
    settings = SimpleNamespace(
        flood_root=tmp_path,
        output_root=tmp_path / "output",
        flood={"weak_max": 4385, "weak_weight": 0.3, "supervision": ["gold", "gold_weak"]},
    )
    manifest = prepare(settings)
    assert manifest["mean"] == [-10, -10]
    assert manifest["normalization_source"] == "gold training chips only"
    assert len(manifest["records"]["weak"]) == 1
    assert manifest["records"]["weak"][0]["weight"] == 0.3


def test_weak_bolivia_and_all_manual_partitions_are_excluded(tmp_path):
    _official_fixture(tmp_path)
    for identifier in ["Bolivia_9", "Ghana_1", "India_2", "Spain_3", "Ghana_5"]:
        _write_pair(tmp_path, identifier, -10, weak=True)
    records, excluded = build_records(tmp_path, 4385)
    assert [record["id"] for record in records["weak"]] == ["Ghana_5"]
    assert excluded["bolivia"] == 1
    assert excluded["gold_overlap"] == 3


def test_loss_weights_chips_instead_of_their_valid_pixel_counts():
    logits = torch.zeros(2, 1, 2, 2)
    target = torch.ones_like(logits)
    target[0, 0, 0] = -1
    weight = torch.ones_like(logits)
    weight[1] = 0.3
    assert segmentation_loss(logits, target, weight, "bce").item() == pytest.approx(np.log(2) * 0.65)


def test_probability_export_preserves_geometry_and_masked_metrics(tmp_path):
    record = _write_pair(tmp_path, "Ghana_1", -10, target=np.array([[-1, 1], [0, 0]]))
    second = _write_pair(tmp_path, "Spain_2", -10, target=np.array([[-1, -1], [-1, 1]]))
    dataset = FloodDataset([record, second], [0, 0], [1, 1])

    class ConstantModel(torch.nn.Module):
        def forward(self, image):
            return torch.zeros((len(image), 1, *image.shape[-2:]), device=image.device)

    output = tmp_path / "evaluation"
    scores = evaluate(ConstantModel(), dataset, torch.device("cpu"), 0.5, output)
    assert scores["mean_chip_iou"] == pytest.approx((1 / 3 + 1) / 2)
    assert scores["pooled_pixels"]["iou"] == pytest.approx(0.5)
    assert scores["pooled_pixels"]["valid_pixels"] == 4
    with rasterio.open(record["image"]) as original, rasterio.open(output / "Ghana_1_probability.tif") as result:
        assert result.crs == original.crs
        assert result.transform == original.transform
        assert result.read(1)[0, 0] == -1
        assert np.all(result.read(1)[1] == 0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
@pytest.mark.parametrize("kind", ["bce", "dice", "combined"])
def test_amp_losses_have_finite_gradients_at_full_chip_size(kind):
    logits = torch.zeros(2, 1, 512, 512, device="cuda", dtype=torch.float16, requires_grad=True)
    target = torch.ones_like(logits, dtype=torch.float32)
    weight = torch.ones_like(target)
    weight[1] = 0.3
    with torch.amp.autocast("cuda"):
        loss = segmentation_loss(logits, target, weight, kind)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_empty_union_is_reported_separately_from_official_smoothed_iou(tmp_path):
    record = _write_pair(tmp_path, "Ghana_1", -10, target=np.zeros((2, 2)))

    class DryModel(torch.nn.Module):
        def forward(self, image):
            return torch.full((len(image), 1, *image.shape[-2:]), -10.0, device=image.device)

    scores = evaluate(
        DryModel(), FloodDataset([record], [0, 0], [1, 1]), torch.device("cpu"), 0.5, tmp_path / "evaluation"
    )
    assert scores["mean_chip_iou"] is None
    assert scores["empty_union_chips"] == 1
    assert scores["reference_mean_chip_iou"] == 1
