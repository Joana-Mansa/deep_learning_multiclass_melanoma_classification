# Validation record

15 September 2026:

- `python main.py --smoke-test --batch-size 4` completed one CPU epoch on 12 generated training images and 12 independent generated validation images.
- The existing ResNet model/training modules were exercised, and the saved checkpoint reloaded successfully with `weights_only=True`.
- NumPy scalar loss values are converted to Python floats so checkpoint metadata remains compatible with weights-only loading.
- No skin-lesion dataset or pretrained weights were downloaded for this check; its metrics are intentionally not reported as project accuracy.

Real-data experiments, advanced ensemble/TTA helpers and patient-level split validation remain separate work requiring the original dataset.
