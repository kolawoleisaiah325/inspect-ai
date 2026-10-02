"""Fit on normal images, calibrate on held-out normals, evaluate once on official test."""
import hashlib
import json
import platform
from importlib.metadata import version
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

from model import Inspector, SIZE, preprocess

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEED = 42

class Features(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.first, self.second = backbone.layer1, backbone.layer2
    def forward(self, image):
        first = self.first(self.stem(image)); second = self.second(first)
        first = F.avg_pool2d(first, 3, 1, 1); second = F.avg_pool2d(second, 3, 1, 1)
        first = F.adaptive_avg_pool2d(first, (SIZE // 8, SIZE // 8))
        return torch.cat([first, second], dim=1)

def pixel_array(path):
    return np.asarray(Image.open(path).convert("RGB").resize((40, 40), Image.Resampling.BILINEAR), np.float32) / 255

def baseline_score(path, mean, scale):
    field = np.sqrt(np.mean(((pixel_array(path) - mean) / scale) ** 2, axis=2))
    return float(np.sort(field.ravel())[-3:].mean()), field

def metrics(rows, threshold, pixel_labels, pixel_scores):
    labels = np.array([r["is_anomaly"] for r in rows]); scores = np.array([r["score"] for r in rows])
    predictions = scores >= threshold
    return {"image_auroc": float(roc_auc_score(labels, scores)),
            "image_average_precision": float(average_precision_score(labels, scores)),
            "pixel_auroc_40x40": float(roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_scores))),
            "threshold": threshold, "true_positive": int(np.sum(predictions & labels)),
            "false_positive": int(np.sum(predictions & ~labels)), "true_negative": int(np.sum(~predictions & ~labels)),
            "false_negative": int(np.sum(~predictions & labels)),
            "median_inference_ms": float(np.median([r["ms"] for r in rows]))}

def main():
    torch.set_num_threads(2); torch.manual_seed(SEED); torch.hub.set_dir(str(ROOT / "cache"))
    rng = np.random.default_rng(SEED); ART.mkdir(exist_ok=True)
    normal = sorted((ROOT / "data/bottle/train/good").glob("*.png"))
    if len(normal) != 209: raise RuntimeError("Expected all 209 official normal training images. Run download_data.py.")
    order = rng.permutation(len(normal)); calibration = [normal[i] for i in order[:42]]; fit = [normal[i] for i in order[42:]]
    network = Features().eval()
    torch.onnx.export(network, torch.zeros(1, 3, SIZE, SIZE), str(ART / "features.onnx"),
                      input_names=["image"], output_names=["features"], opset_version=17, dynamo=False)
    # Check export against the exact frozen PyTorch extractor before fitting.
    import onnxruntime as ort
    session = ort.InferenceSession(str(ART / "features.onnx"), providers=["CPUExecutionProvider"])
    check = preprocess(Image.open(fit[0]).convert("RGB"))
    with torch.inference_mode(): expected = network(torch.from_numpy(check)).numpy()
    actual = session.run(None, {"image": check})[0]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    feature_batches = []
    with torch.inference_mode():
        for start in range(0, len(fit), 8):
            batch = np.concatenate([preprocess(Image.open(p).convert("RGB")) for p in fit[start:start+8]])
            feature_batches.append(network(torch.from_numpy(batch)).numpy())
            print(f"Features {min(start+8,len(fit))}/{len(fit)}", flush=True)
    full = np.concatenate(feature_batches).transpose(0, 2, 3, 1).reshape(-1, 192)
    scale = np.maximum(full.std(axis=0), .05).astype(np.float32)
    # Stratified random patches: each fitted image contributes equally. No coreset claim.
    feature_images = full.reshape(len(fit), -1, 192)
    bank = np.concatenate([image[rng.choice(len(image), 24, replace=False)] for image in feature_images]) / scale
    np.savez_compressed(ART / "memory.npz", bank=bank.astype(np.float32), scale=scale)
    metadata = {"model_version": "inspectai-bottle-v1", "backbone": "ResNet18 ImageNet1K V1 layers 1+2",
                "input_size": SIZE, "feature_channels": 192, "patch_grid": [20,20], "memory_patches": len(bank),
                "fit_images": len(fit), "calibration_images": len(calibration), "seed": SEED,
                "threshold": 1.0, "heatmap_ceiling": 1.0,
                "threshold_rule": "95th percentile of image scores on 42 held-out normal training images",
                "image_score": "Mean of the three largest nearest-normal-patch distances",
                "dataset": "MVTec AD bottle", "dataset_license": "CC BY-NC-SA 4.0",
                "mirror_revision": "c75b39616f84db43677bcc8228caaafaf5096d7f"}
    (ART / "metadata.json").write_text(json.dumps(metadata, indent=2))
    inspector = Inspector(ART)
    calibration_scores, calibration_fields = [], []
    for path in calibration:
        score, field = inspector.score_image(Image.open(path).convert("RGB"))
        calibration_scores.append(score); calibration_fields.append(field.ravel())
    metadata["threshold"] = float(np.quantile(calibration_scores, .95))
    metadata["heatmap_ceiling"] = float(np.quantile(np.concatenate(calibration_fields), .995) * 1.8)
    metadata["calibration_false_alarms"] = int(np.sum(np.array(calibration_scores) >= metadata["threshold"]))
    metadata["files_sha256"] = {name: hashlib.sha256((ART/name).read_bytes()).hexdigest() for name in ["features.onnx","memory.npz"]}
    (ART / "metadata.json").write_text(json.dumps(metadata, indent=2)); inspector.metadata = metadata
    pixels = np.stack([pixel_array(p) for p in fit]); mean = pixels.mean(axis=0); pixel_scale = np.maximum(pixels.std(axis=0), .025)
    baseline_threshold = float(np.quantile([baseline_score(p, mean, pixel_scale)[0] for p in calibration], .95))
    np.savez_compressed(ART / "baseline.npz", mean=mean, scale=pixel_scale, threshold=baseline_threshold)
    test = sorted((ROOT / "data/bottle/test").glob("*/*.png"))
    if len(test) != 83: raise RuntimeError("Expected all 83 official test images.")
    records = {"pixel_baseline": [], "cnn_patch_memory": []}; masks = []; fields = {k:[] for k in records}
    for count, path in enumerate(test, 1):
        anomaly = path.parent.name != "good"
        if anomaly:
            maskpath = ROOT / "data/bottle/ground_truth" / path.parent.name / f"{path.stem}_mask.png"
            mask = np.asarray(Image.open(maskpath).convert("L").resize((40,40), Image.Resampling.NEAREST)) > 0
        else: mask = np.zeros((40,40), bool)
        masks.append(mask.ravel())
        for method in records:
            started = time.perf_counter()
            if method == "pixel_baseline": score, field = baseline_score(path, mean, pixel_scale)
            else: score, field = inspector.score_image(Image.open(path).convert("RGB"))
            ms = (time.perf_counter() - started) * 1000
            records[method].append({"path": str(path.relative_to(ROOT/"data")).replace("\\","/"), "is_anomaly": anomaly, "score": score, "ms": round(ms,3)})
            if field.shape != (40,40): field = np.asarray(Image.fromarray(field.astype(np.float32)).resize((40,40), Image.Resampling.BILINEAR))
            fields[method].append(field.ravel())
        print(f"Evaluated {count}/{len(test)}", flush=True)
    result = {"dataset": "MVTec AD bottle", "split": {"fit": len(fit), "normal_calibration": len(calibration), "test": len(test), "test_normal":20,"test_anomaly":63},
              "seed":SEED, "primary_method_selected_before_test":"cnn_patch_memory", "fit_paths":[p.name for p in fit], "calibration_paths":[p.name for p in calibration],
              "methods": {name:{"metrics": metrics(rows, baseline_threshold if name=="pixel_baseline" else metadata["threshold"], masks, fields[name]), "records": rows} for name,rows in records.items()},
              "notes": ["Frozen pretrained CNN, not fine-tuned on defects.","Thresholds use held-out normal training images only.","The primary method and settings were fixed before reading test results.","Pixel AUROC is evaluated on masks resized to 40x40; it is not an official full-resolution benchmark metric.","Latency is a local CPU measurement, including file decode, excluding server startup and network.","A single category benchmark does not establish performance on camera uploads."]}
    result["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "processor": platform.processor(),
                             "packages": {name: version(name) for name in ["torch", "torchvision", "numpy", "Pillow", "onnx", "onnxruntime", "scikit-learn"]}}
    (ART / "evaluation.json").write_text(json.dumps(result, indent=2))
    public = ROOT / "public"; (public/"samples").mkdir(parents=True, exist_ok=True)
    samples = []
    for label, name, filename in [("normal","good","000.png"),("large break","broken_large","000.png"),("small break","broken_small","000.png"),("contamination","contamination","000.png")]:
        path = ROOT/"data/bottle/test"/name/filename
        identifier = name+"-000"
        target = public/"samples"/(identifier+".jpg")
        image = Image.open(path).convert("RGB"); image.resize((512,512),Image.Resampling.LANCZOS).save(target,quality=92)
        # This is a resized demonstration derivative, not the image used for evaluation.
        output = inspector.inspect(target.read_bytes())
        (public/"samples"/(identifier+"-heat.png")).write_bytes(__import__('base64').b64decode(output.pop("heatmap").split(',')[1]))
        samples.append({"id":identifier,"label":label,"file":"samples/"+identifier+".jpg","source_path":str(path.relative_to(ROOT/"data")).replace("\\","/"),"recorded_result":output})
    (public/"samples.json").write_text(json.dumps(samples,indent=2))
    (public/"evaluation.json").write_text(json.dumps(result,indent=2))
    shutil.copy(ROOT/"data/bottle/license.txt", public/"DATA-LICENSE.txt")
    print(json.dumps({name:data["metrics"] for name,data in result["methods"].items()},indent=2),flush=True)

if __name__ == "__main__": main()
