# 🔬 Three-Class Skin Lesion Classification

Joana Owusu-Appiah’s academic image-classification project for **melanoma (MEL), basal cell carcinoma (BCC) and squamous cell carcinoma (SCC)**. The maintained entry point trains a ResNet-50 using the existing model and training modules.

## Quickstart

```bash
git clone https://github.com/Joana-Mansa/deep_learning_multiclass_melanoma_classification.git
cd deep_learning_multiclass_melanoma_classification
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --smoke-test
```

The smoke test trains for one CPU epoch on generated images, reloads the saved checkpoint and writes `outputs/training_summary.json`. It verifies the code path; its scores are not skin-cancer results. No dataset or pretrained weights are downloaded.

## Train on your prepared dataset

```bash
python main.py --train-dir data/train --val-dir data/val --epochs 30
```

Each directory must contain `mel/`, `bcc/` and `scc/` image folders. Class indices remain MEL=0, BCC=1, SCC=2. Keep related images/patients in the same split; the command checks directory separation but cannot establish patient independence without metadata. ImageNet weights download on first use; `--no-pretrained` disables that download.

## Repository map

| File | Purpose |
|---|---|
| `main.py` | Maintained single-model training CLI and generated-data check |
| `models.py` | ResNet and alternative model definitions |
| `train.py` | Training, validation, augmentation and checkpoint helpers |
| `dataset.py`, `data_utils.py` | Transforms and original data-preparation helpers |
| `test.py` | Prediction export and experimental evaluation/ensemble routines; not a unit-test suite |
| `config.py` | Original experiment defaults and class definitions |

📖 [Workflow and validation status](docs/workflow.md)

## Results and scope

Real-data training has not been reproduced in this maintenance pass; no new performance claim is made. The older exploratory main script remains in Git history. Its hard-coded paths and undefined calls have been replaced by the documented CLI. Alternative backbones, ensembles, TTA and cross-validation helpers remain research extensions requiring separate end-to-end validation.

**Joana Owusu-Appiah** · [LinkedIn](https://www.linkedin.com/in/joana-owusu-appiah-msc-8751a9166/)
