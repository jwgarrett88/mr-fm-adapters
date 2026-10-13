# MR Foundation Model Adapters

This repository contains training and inference code for MRI classification
tasks using frozen foundation-model embeddings and 3D DenseNet models. The
tasks covered are anatomical region, sequence, contrast, and fat-suppression
classification.

## Repository contents

- `EmbeddingClassificationTrainingWithOptuna.py` trains an MLP classifier on
  precomputed embeddings and tunes its hyperparameters with Optuna.
- `DenseNetModelTraining/` contains task-specific training and inference
  notebooks.
- The task folders currently include trained `.pth` model checkpoints.

## Data and privacy

No MRI images, metadata CSVs, or embedding arrays are provided. You must have
authorized access to the data and any required model inputs before running the
notebooks. The notebooks may connect to Flywheel when their download cells are
run; configure credentials through your approved local environment rather than
putting tokens in notebook source.

Notebook paths can be configured with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MRI_DATA_ROOT` | `data` | Root directory for local image/data files. |
| `MRI_METADATA_CSV` | `$MRI_DATA_ROOT/filtered_df_for_share.csv` | Metadata CSV path. |
| `MRI_MODEL_DIR` | `.` | Directory used to load and save model checkpoints. |
| `MRI_OUTPUT_DIR` | `outputs` | Directory for generated prediction CSVs. |

The metadata CSV must contain the columns required by the workflow being run.
The notebooks use fields such as `flywheel_info.nifti_name` and task labels;
the embedding script requires `flywheel_info.nifti_path`, task labels, and a
split column. Set `MRI_DATA_ROOT` and `MRI_METADATA_CSV` to locations
authorized for your environment; the defaults are placeholders, not bundled
data.

For example, in PowerShell:

```powershell
$env:MRI_DATA_ROOT = "D:\authorized-data\images"
$env:MRI_METADATA_CSV = "D:\authorized-data\metadata.csv"
$env:MRI_MODEL_DIR = "D:\authorized-models"
$env:MRI_OUTPUT_DIR = "D:\authorized-results"
```

Do not publish clinical images, metadata, embeddings, prediction files, logs,
credentials, or other private run artifacts. Notebook outputs have been
cleared from the checked-in notebooks. Clear outputs again before sharing any
notebook that you execute locally. Model checkpoints are not included in this
project and are not available for redistribution. The local checkpoints were
trained on de-identified MR images; access to those local files does not grant
permission to share them.
Inference notebooks require a separately obtained checkpoint. Set
`MRI_MODEL_DIR` to the local directory containing the appropriate task's
checkpoint before running an inference notebook.

Generated data and result files are covered by `.gitignore`; review `git
status` and inspect the complete repository before making it public.

## Environment

No dependency lockfile or tested version matrix is provided. The Python
training script uses PyTorch, NumPy, pandas, scikit-learn, Matplotlib,
seaborn, Optuna, and tqdm. The DenseNet notebooks additionally use MONAI,
nibabel, and TensorBoard; notebooks that download from Flywheel also require
the organization's approved Flywheel client and authentication setup.

Create an isolated Python environment and install compatible versions of the
dependencies for your platform. Install PyTorch using the instructions for
your CPU/GPU configuration. The notebooks were authored as Jupyter notebooks;
run them with a kernel that has the required packages installed.

## Train an embedding classifier

The script expects a metadata CSV and matching precomputed embedding arrays.
For each `flywheel_info.nifti_path`, it looks for an array whose filename is
formed by replacing `.nii.gz` with `_GE_emb_11220225.npy`. It also expects
task labels and split information in the CSV. Input data and embeddings are
not included.

Example:

```powershell
python .\EmbeddingClassificationTrainingWithOptuna.py `
  --ratio 0.25 `
  --csv_file "D:\authorized-data\metadata.csv" `
  --save_subfolder "sequence-025" `
  --tag sequence
```

`--tag` accepts `sequence`, `contrast`, `fat_suppression`, or `region`.
The script writes cached arrays, logs, and experiment results beneath
`MRI_Anatomy_Cohort_metadata/` in the current working directory.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE). Confirm
you are authorized to distribute the code and included model checkpoints under
this license before public release.
