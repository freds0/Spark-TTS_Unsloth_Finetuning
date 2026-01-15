"""
Module to load and manage YAML configurations for the Spark-TTS project.
"""

import os
import yaml
from typing import Any, Dict, Optional
import torch


class Config:
    """Class to load and access configurations from YAML file."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration loader.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._process_auto_values()

    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def _process_auto_values(self):
        """Process 'auto' values in configurations."""
        # Process fp16 and bf16
        if self.config["training"]["fp16"] == "auto":
            self.config["training"]["fp16"] = not torch.cuda.is_bf16_supported()

        if self.config["training"]["bf16"] == "auto":
            self.config["training"]["bf16"] = torch.cuda.is_bf16_supported()

    def get(self, *keys, default=None) -> Any:
        """
        Get a value from the configuration dictionary using nested keys.

        Args:
            *keys: Sequence of keys to access nested values.
            default: Default value if the key is not found.

        Returns:
            The configuration value or the default value.

        Example:
            config.get("data", "audio", "target_sampling_rate")
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Allow direct access using config['key']."""
        return self.config[key]

    def __contains__(self, key: str) -> bool:
        """Allow checking if a key exists using 'key in config'."""
        return key in self.config

    # Convenience methods to access specific sections
    @property
    def data(self) -> Dict[str, Any]:
        """Return data configurations."""
        return self.config["data"]

    @property
    def model(self) -> Dict[str, Any]:
        """Return model configurations."""
        return self.config["model"]

    @property
    def lora(self) -> Dict[str, Any]:
        """Return LoRA configurations."""
        return self.config["lora"]

    @property
    def training(self) -> Dict[str, Any]:
        """Return training configurations."""
        return self.config["training"]

    @property
    def audio_tokenizer(self) -> Dict[str, Any]:
        """Return audio tokenizer configurations."""
        return self.config["audio_tokenizer"]

    @property
    def synthesis(self) -> Dict[str, Any]:
        """Return synthesis configurations."""
        return self.config["synthesis"]

    @property
    def system(self) -> Dict[str, Any]:
        """Return system configurations."""
        return self.config["system"]

    def update(self, section: str, key: str, value: Any):
        """
        Update a specific value in the configuration.

        Args:
            section (str): Configuration section (e.g., 'training').
            key (str): Key to be updated.
            value (Any): New value.
        """
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
        else:
            raise KeyError(f"Key '{section}.{key}' not found in configuration.")

    def save(self, output_path: Optional[str] = None):
        """
        Save the current configuration to a YAML file.

        Args:
            output_path (str): Path to save. If None, overwrites the original file.
        """
        save_path = output_path or self.config_path

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

        print(f"Configuration saved to: {save_path}")

    def display(self, section: Optional[str] = None):
        """
        Display the configurations in a formatted way.

        Args:
            section (str): Specific section to display. If None, displays everything.
        """
        import json

        if section:
            if section in self.config:
                print(json.dumps({section: self.config[section]}, indent=2))
            else:
                print(f"Section '{section}' not found.")
        else:
            print(json.dumps(self.config, indent=2))


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Helper function to load the configuration.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Config: Loaded configuration object.
    """
    return Config(config_path)


# Exemplo de uso
if __name__ == "__main__":
    # Carregar configuração
    config = load_config()

    # Acessar configurações
    print("Configurações de Dados:")
    print(f"  Dataset path: {config.data['local_dataset_path']}")
    print(f"  Sampling rate: {config.get('data', 'audio', 'target_sampling_rate')}")

    print("\nConfigurações de Treinamento:")
    print(f"  Batch size: {config.training['per_device_train_batch_size']}")
    print(f"  Learning rate: {config.training['learning_rate']}")

    print("\nConfigurações de LoRA:")
    print(f"  Enabled: {config.lora['enabled']}")
    print(f"  Rank: {config.lora['r']}")

    # Exibir seção específica
    print("\n" + "=" * 50)
    config.display("model")
