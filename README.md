
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

4. **Create an env file:**

   Create a `.env` file in the root directory of the project to store your environment variables. This file should include your Wandb API key, HF token and any other sensitive information. For example:

   ```bash
   
   touch .env


   ```
   After creating the file, you must follow the instructions in the `.env.example` file to set up your environment variables. This is crucial for the project to run correctly, especially for features like Wandb logging and Hugging Face model access.

   ```bash

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
    python main.py +user=<your_name> +encoder@experiment.encoder=resnet18
    ```

**2. Editing Configuration Files (for permanent changes)**

To change the default behavior of the project, you can directly edit the YAML files in the `configs` directory. For instance, to permanently change the default configs, open `configs/experiment/base_module.yaml`. This approach is ideal for changes you want to keep for future runs.

### Hyperparameter Sweeping

Hydra's sweeping functionality allows you to run multiple experiments with different hyperparameter combinations. To perform a sweep over different encoders, you can use the following command:

```bash
python main.py -m +user=<your_name> +encoder@experiment.encoder=resnet18,vit16b
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
└── .env                    # Environment variables
```
