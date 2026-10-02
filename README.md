# InspectAI

Bottle anomaly inspection by Isaiah Kolawole. A working FastAPI model endpoint,
an original/heatmap comparison, and a benchmark with a fixed normal-only threshold.
This is a **noncommercial research prototype**, not an industrial quality guarantee.

[Live inspection workspace](https://isaiah-inspect-ai.vercel.app/) ·
[Portfolio case study](https://isaiah-kolawole-portfolio.vercel.app/inspect-ai.html)

## Run locally

Python 3.12, from this directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8003
```

Open http://127.0.0.1:8003. Exported model artifacts and four attributed samples are
included, so running the app needs no dataset download or training dependency.
The initial sample view explicitly shows a **recorded** result. Run live inspection
to call the real model API. User uploads are processed in memory, never written to disk.
History keeps only review metadata in browser local storage; clear or export it from the UI.

## Model and evaluation

The frozen ImageNet-pretrained ResNet18 supplies layer 1 and 2 features. Features
are pooled to a 20x20 grid and normalized by the fitted normal-feature standard
deviation. Each fitted normal image contributes 24 random patches to a memory bank.
Each inspection patch is scored by Euclidean distance to its nearest normal patch.
The image score averages the three largest patch distances.

This is a small, independently implemented **patch-memory model**, inspired by
normal-feature methods such as [PatchCore](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html).
It uses random stratified sampling, not greedy coreset selection, and does not
reproduce the paper's architecture, scores or results.

Of 209 official normal training images, 167 fit the memory and 42 set the 95th
percentile review threshold (seed 42). Both this method and a pixel-statistics
baseline are scored on all 83 official test images: 20 normal and 63 anomalous.
The primary method and settings were specified before examining test results.
Raw per-image scores, confusion counts and file names are in `artifacts/evaluation.json`.
No defect test images fit the model or set the threshold.

Recorded CNN image AUROC: **0.9976**; pixel baseline: **0.9698**. At the normal-calibrated threshold, the CNN misses 2/63 defects and flags 0/20 normal images. The baseline misses 12/63 and flags 0/20. See [the full report](docs/evaluation.md), including the two contamination misses.

Anomaly scores are **not probabilities**. A fixed heatmap color scale comes from
normal calibration patches. It does not identify a defect type or provide an
official segmentation mask. Pixel AUROC uses downsampled 40x40 annotations.
The 512px JPEG demo derivatives are distinct from the original test images.
New backgrounds, products and camera views are outside the measured scope.

## Reproduce the model

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe train.py
```

`download_data.py` downloads only the bottle category from the pinned mirror
`foersben/mvtec-ad`, revision `c75b39616f84db43677bcc8228caaafaf5096d7f`,
and records SHA-256 hashes. Dataset and cache directories are ignored by Git.
Training exports the CNN as ONNX and checks its numerical agreement with PyTorch.
Serving uses ONNX Runtime and NumPy, without torch, sklearn or the full dataset.

## API

- `GET /api/health`: model version and metadata.
- `POST /api/inspect`: raw JPEG/PNG/WebP bytes; maximum 2 MB and 16 megapixels.
  Returns score, threshold, review signal, fixed-scale PNG heatmap and inference timing.
- `/docs`: interactive OpenAPI documentation.

```powershell
curl.exe -X POST http://127.0.0.1:8003/api/inspect -H "Content-Type: image/jpeg" --data-binary "@public/samples/broken_large-000.jpg"
```

Tests run without torch, network access or the full dataset:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`Dockerfile` supplies a non-root local container configuration. A Docker build
requires a Docker installation; the configuration itself is not proof of a tested container.
Vercel detects the FastAPI entry point `app.py`; its static assets are in `public/`.
The root URL redirects to the CDN-served `index.html`. Local static-file mounting
is conditional because Vercel omits `public/` from the Python function bundle.
No auth or persistent image storage is configured for this bounded public demo.
An industrial service would need operational review, access controls, and independent validation.

## Attribution and licensing

MVTec AD: copyright 2019 MVTec Software GmbH. Dataset, derived sample images and
normal-feature memory are provided under **CC BY-NC-SA 4.0**. See
`public/DATA-LICENSE.txt` and the [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
Displayed images are resized JPEG derivatives of the original PNG files; source
paths are recorded in `public/samples.json`. Do not use these assets commercially.

Cite: Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger,
*MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
The frozen feature extractor derives from torchvision ResNet18 ImageNet1K V1 weights
at `https://download.pytorch.org/models/resnet18-f37072fd.pth`; torchvision uses a
BSD 3-Clause license. Application source code is MIT; third-party assets retain their terms.
