# food_classifier

Classifies food items from the Food 101 dataset using PyTorch and Lightning.\
[Link to assignment](assignment.md)

## Configured Run Jobs

- `poetry run main` runs the `main()` function in `main.py`, which starts the training as specified in `config.yaml`.
- `poetry run experiments` runs the `main_exp()` function in `experiments/experiments.py`. Here, test images are classified and the results are plotted.
- `poetry run tensorboard --logdir logs` starts the `tensorboard` server. Access it in your browser on `[localhost](http://localhost:6006/)`.

## What can be improved?

- Confusion matrix is too small and filename duplicates could happen
- Tensorboard integration and metrics can be improved
- Checkpoints are not used

## Quick Poetry Usage Guide

Some quick and useful commands:

- `poetry shell` starts the virtual environment.
- `exit` lets you leave the virtual environment again.
- `poetry add` lets you add a new dependency.
- `poetry install` lets you install your dependencies and/or change your configuration. Do this every time you added a new dependency or if you edited the `pyproject.toml` file.
