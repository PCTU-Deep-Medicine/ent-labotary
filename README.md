
## Features

- **PyTorch Lightning:** For clear and organized model training and evaluation.
- **Hydra:** For flexible and dynamic configuration management.
- **Wandb:** For experiment tracking and visualization.

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Logging in to Weights & Biases:**

   Before running any experiments, you need to log in to your Weights & Biases account. Run the following command and follow the instructions:

   ```bash
   wandb login
   ```

5. **Install pre-commit hook:**
   
   This is required for contributors
   ```bash
   pre-commit install
   ```

## Usage

### User Name Requirement

The `+user=<your_name>` argument is **required** for all commands. This is used to identify you in experiment tracking logs.

### Running the project

The main entry point for this project is `main.py`. You can run it directly with the default configuration:

```bash
python main.py +user=<your_name>
```

### Managing Configuration

You can manage the project's configuration in two ways:

**1. Command-Line Overrides (for temporary changes)**

For one-off experiments or quick tests, you can override any configuration parameter directly from the command line using Hydra for example:

*   **Run with a different encoder:**
    ```bash
    python main.py +user=<your_name> encoder=resnet18
    ```

*   **Use the MNIST datamodule:**
    ```bash
    python main.py +user=<your_name> datamodule=mnist
    ```

*   **Change the max epochs:**
    ```bash
    python main.py +user=<your_name> experiment.trainer.max_epochs=5
    ```

**2. Editing Configuration Files (for permanent changes)**

To change the default behavior of the project, you can directly edit the YAML files in the `configs` directory. For instance, to permanently change the default configs, open `configs/experiment/base_module.yaml`. This approach is ideal for changes you want to keep for future runs.

### Hyperparameter Sweeping

Hydra's sweeping functionality allows you to run multiple experiments with different hyperparameter combinations. To perform a sweep over different encoders, you can use the following command:

```bash
python main.py +user=<your_name> --multirun encoder=resnet18,vit16b
```

This will run two experiments, one with the `resnet18` encoder and another with the `vit16b` encoder.

## Configuration

The configurations for this project are managed by Hydra and are located in the `configs` directory.

- `config.yaml`: The main configuration file that composes the other configuration files.
- `callbacks/`: Configuration for PyTorch Lightning callbacks.
- `datamodule/`: Configuration for the data modules (e.g., `mnist.yaml`, `default.yaml`).
- `encoder/`: Configuration for the encoder models (e.g., `resnet18.yaml`, `vit16b.yaml`).
- `experiment/`: Configuration for the experiments.
- `logger/`: Configuration for the logger (e.g., `wandb.yaml`).
- `loss/`: Configuration for the loss functions.
- `trainer/`: Configuration for the PyTorch Lightning Trainer.

## Project Structure

```
├── configs/                # Hydra configuration files
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter notebooks for exploration
├── outputs/                # Outputs from experiments
├── scripts/                # Helper scripts
├── src/                    # Source code
│   ├── components/         # Reusable components
│   ├── datamodule/         # Data modules
│   ├── experiment/         # Experiment modules
│   └── utils/              # Utility functions
├── .gitignore              # Git ignore file
├── main.py                 # Main entry point
├── README.md               # This file
└── requirements.txt        # Project dependencies
```
