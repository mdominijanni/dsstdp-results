# Delay Learning Experiments on Diehl & Cook using Inferno
## How to Run
The project can be run by directly executing the included `ddc` module. This includes three subprojects named `dc`, `drdc`, and `dsdc` with slightly different configuration specifications. All train the same underlying Diehl & Cook model, but trained using STDP, [DR-STDP](https://direct.mit.edu/neco/article-abstract/36/7/1332/121125/Bioplausible-Unsupervised-Delay-Learning-for), and [DS-STDP](https://www.proquest.com/dissertations-theses/investigations-into-simulation-training-spiking/docview/3167831788/se-2) respectively. Not directly tied to the filename, each configuration includes a field `meta.name` which can be used to restore a run.

### Examples
#### STDP, Create a New Run
`python -m ddc dc configs/dc-baseline-n100.toml`

#### DR-STDP, Restore a Run from the Latest Checkpoint
`python -m ddc drdc configs/drdc-baseline-n100.toml -r drdc-baseline-n100/latest.pt`

#### DS-STDP, Restore a Run from the End of Epoch 11
`python -m ddc dsdc configs/dsdc-baseline-n100.toml -r dsdc-baseline-n100/epochfinal.11.pt`


## Included Files
### Main Codebase
The source code is split into three main folders.
- `inferno`: main public library code, including some fixes past the currently published PyPI version (the full repository, including a link to the documentation can be found [here](https://github.com/mdominijanni/inferno)).
- `firebrand`: experimental extensions to `inferno` not yet made public or integrated into the main library.
- `ddc`: module containing the experiments.

### Configurations
Configurations are specified via TOML files. Three are included at the top level and replicate the baseline 100 neuron results from the dissertation.
- `dc-baseline-n100.toml`: Training with STDP.
- `drdc-baseline-n100.toml`: Training with [Delay-Related STDP (DR-STDP)](https://direct.mit.edu/neco/article-abstract/36/7/1332/121125/Bioplausible-Unsupervised-Delay-Learning-for).
- `dsdc-baseline-n100.toml`: Training with [Delay-Shifted STDP (DS-STDP)](https://www.proquest.com/dissertations-theses/investigations-into-simulation-training-spiking/docview/3167831788/se-2).

Additionally, the directories `configs/dc`, `configs/drdc`, and `configs/dsdc` contain the configurations used in the paper for generating results using STDP, DR-STDP, and DS-STDP, respectively. The only change made to these configurations files from the data generating runs was setting the device to `"cuda"` (rather than targeting a specific CUDA device).

### Results
The `results` directory contains the raw results from WandB in the `.csv` files, and that data aggregated into `.pickle` files. To regenerate those files locally, `stdp-metrics.py`, `drstdp-metrics.py`, and `dsstdp-metrics.py` can be rerun from inside the same directory as the corresponding `.csv` files. The notebook `review.ipynb` contains code to generate the figures and statistics used in the paper from the `.pickle` files.

### Additional Files
`workspaces.py`: includes function(s) to create WandB workspace views. They require providing a WandB entity name (either a username or a team name), along with the name of the project.

### Requirements
A minimum of Python 3.11 should be required, although testing has only been conducted using Python 3.12 and 3.13. Two (mostly) minimal `requirements.txt` compatible files are included, one for CPU only execution and one with CUDA support. `wandb-workspaces` is included to programmatically configure Weights & Biases workspaces.
- `requirements-cpu.txt`: minimal required packages, without CUDA support.
- `requirements-cuda.txt`: minimal required packages, with CUDA 12.6 support.
