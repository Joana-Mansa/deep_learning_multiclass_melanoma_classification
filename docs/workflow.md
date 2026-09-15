# Training workflow

## Data and labels

Obtain the original course data separately; no verified public download for that exact split is provided here. Training and validation use class-organized image folders. The maintained CLI maps alphabetical folder discovery back to the original class convention: MEL 0, BCC 1, SCC 2.

Prepare splits before augmentation, keeping each patient/lesion within a single split. Preserve source identifiers to audit overlap. The command does not create or certify a patient-level split. Validation uses deterministic resize/normalization; training adds augmentation.

## Outputs

`outputs/best_model.pt` stores the best validation-accuracy checkpoint. `training_summary.json` records data provenance, class mapping, seed, sample counts, metrics and history. The checkpoint is reloaded to verify serialization. The first epoch is eligible for saving even if validation accuracy is zero.

The default maintained route is single-model ResNet training. It uses existing `models.py` and `train.py`, not a separate illustrative classifier. Model alternatives, ensemble and cross-validation code in the repository are retained but are not presented as tested CLI modes.

## Repairs

- Fixed the `training` versus `train.py` import mismatch.
- Replaced the broken hard-coded main routine with explicit arguments and a downloadable-weight opt-out.
- Removed an unsupported scheduler argument and made checkpoint creation robust to zero initial validation accuracy.
- Labelled image-loading errors now raise instead of silently training on a dummy image assigned to class 0.
- README paths and examples now match tracked files; placeholder clone/contact URLs were corrected.

## Verification

See `validation.md` for the commands actually run. A generated-data check establishes execution and checkpoint behavior, not scientific accuracy. Full dataset experiments, advanced helper modes and patient-independent evaluation remain separate verification tasks.
