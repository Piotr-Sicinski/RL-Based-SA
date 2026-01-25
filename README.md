# Reinforcement Learning Based Simulated Annealing

Implementation by Mikołaj Roszczyk and Piotr Siciński.

Original paper:

**Nathan Qiu, Daniel Liang. "Reinforcement Learning Based Simulated Annealing".** Stony Brook University, USA.


## How to install

Make sure to have Python ≥3.10 (tested with Python 3.10.11) and 
ensure the latest version of `pip` (tested with 22.3.1):
```bash
pip install --upgrade --no-deps pip
```

Next, install PyTorch 1.13.0 with the appropriate CUDA version (tested with CUDA 11.7):
```bash
python -m pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Finally, install the remaining dependencies using pip:
```bash
pip install -r requirements.txt
```

To run the code, the project root directory needs to be added to your PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

## Running experiments
### Training
The main run file to reproduce all experiments is `main.py`. We use [Hydra](https://hydra.cc/) to configure experiments, so you can retrain our models as follows
```bash
python scripts/main.py +experiment=<config_file>

python scripts/main.py +experiment=knapsack_nsa_ppo
python scripts/main.py +experiment=rosenbrock_rlbsa_ppo
```

### Evaluation

```bash
python scripts/simple_eval.py
```

Configuration file `scripts/conf/simple_eval.yaml` allows for choosing methodes and parameters to compare.

