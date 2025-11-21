#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Manage command for Models.

This module provides an interactive CLI for setting up the model configuration
without requiring users to manually write conf/agent.yml files.
"""

from getpass import getpass
from typing import Dict

from rich.console import Console
from rich.prompt import Confirm, Prompt

from datus.configuration.agent_config import ModelConfig, load_model_config
from datus.configuration.agent_config_loader import configuration_manager
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
console = Console()


def _validate_model_name(name: str) -> tuple[bool, str]:
    """Validate model name format."""
    if not name.strip():
        return False, "Model name cannot be empty"
    # Check for invalid characters that could cause issues in YAML
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " ", "\t", "\n"]
    for char in invalid_chars:
        if char in name:
            return False, f"Model name cannot contain '{char}'"
    return True, ""


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL format."""
    if not url.strip():
        return False, "URL cannot be empty"
    if not (url.startswith("http://") or url.startswith("https://")):
        return False, "URL must start with http:// or https://"
    return True, ""


class ModelsManager:
    def __init__(self, config_path: str):
        try:
            # Use ConfigurationManager directly to avoid namespace validation
            self.config_manager = configuration_manager(config_path=config_path)
            self.models: Dict[str, ModelConfig] = {}
            self.target: str = ""

            # Load models from configuration
            models_raw = self.config_manager.get("models", {})
            if models_raw:
                self.models = {name: load_model_config(cfg) for name, cfg in models_raw.items()}

            # Load target model
            self.target = self.config_manager.get("target", "")

            self.config_loaded = True
        except DatusException as e:
            if e.code == ErrorCode.COMMON_FILE_NOT_FOUND:
                console.print("❌ Configuration file not found.")
                console.print("Please run 'datus-agent init' first to create the configuration.")
                console.print("Or specify a config file with --config <path>")
            else:
                console.print(f"❌ {e.message}")
            self.config_loaded = False
        except Exception as e:
            console.print(f"❌ Failed to load configuration: {e}")
            self.config_loaded = False

    def run(self, command: str) -> int:
        """Run the specified models command."""
        if not self.config_loaded:
            return 1

        if command == "list":
            return self.list()
        elif command == "add":
            return self.add()
        elif command == "delete":
            return self.delete()
        elif command == "set-target":
            return self.set_target()
        else:
            console.print(f"❌ Unknown command: {command}")
            return 1

    def list(self) -> int:
        """List all configured models."""
        if not self.models:
            console.print("No models configured.")
            return 0

        console.print("[bold yellow]Configured models:[/bold yellow]")
        console.print(f"Target model: [bold green]{self.target}[/bold green]\n")

        for model_name, model_config in self.models.items():
            is_target = " [bold green](active)[/bold green]" if model_name == self.target else ""
            console.print(f"Model: {model_name}{is_target}")
            console.print(f"  Type: {model_config.type}")
            console.print(f"  Base URL: {model_config.base_url}")
            console.print(f"  Model: {model_config.model}")
            console.print(f"  Save LLM Trace: {model_config.save_llm_trace}")
            console.print(f"  Enable Thinking: {model_config.enable_thinking}")
            # Don't print API key for security
            console.print(f"  API Key: {'*' * 8 if model_config.api_key else '(not set)'}")
            console.print()
        return 0

    def add(self) -> int:
        """Interactive method to add a new model configuration."""
        console.print("[bold yellow]Add New Model[/bold yellow]")

        # Model name
        model_name = Prompt.ask("- Model name (e.g., openai, anthropic, deepseek)")
        valid, error_msg = _validate_model_name(model_name)
        if not valid:
            console.print(f"❌ {error_msg}")
            return 1

        # Check if model already exists
        if model_name in self.models:
            console.print(f"❌ Model '{model_name}' already exists")
            return 1

        # Model type selection
        model_types = ["openai", "claude", "gemini", "deepseek", "qwen"]
        model_type = Prompt.ask("- Model type", choices=model_types, default="openai")

        # Base URL configuration
        default_urls = {
            "openai": "https://api.openai.com/v1",
            "claude": "https://api.anthropic.com",
            "gemini": "https://generativelanguage.googleapis.com/v1beta",
            "deepseek": "https://api.deepseek.com",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        default_url = default_urls.get(model_type, "https://api.example.com")
        base_url = Prompt.ask("- Base URL", default=default_url)
        valid, error_msg = _validate_url(base_url)
        if not valid:
            console.print(f"❌ {error_msg}")
            return 1

        # API Key
        api_key = getpass("- API Key: ")
        if not api_key.strip():
            console.print("❌ API Key cannot be empty")
            return 1

        # Model name/identifier
        default_models = {
            "openai": "gpt-4o-mini",
            "claude": "claude-haiku-4-5",
            "gemini": "gemini-2.0-flash-exp",
            "deepseek": "deepseek-chat",
            "qwen": "qwen3-coder-plus",
        }
        default_model = default_models.get(model_type, "")
        model = Prompt.ask("- Model identifier", default=default_model)
        if not model.strip():
            console.print("❌ Model identifier cannot be empty")
            return 1

        # Optional settings
        save_llm_trace = Confirm.ask("- Save LLM trace?", default=False)
        enable_thinking = Confirm.ask("- Enable thinking mode?", default=False)

        # Create model configuration
        model_config = ModelConfig(
            type=model_type,
            base_url=base_url,
            api_key=api_key,
            model=model,
            save_llm_trace=save_llm_trace,
            enable_thinking=enable_thinking,
        )

        # Add to configuration
        self.models[model_name] = model_config

        # Ask if this should be the target model
        if not self.target or Confirm.ask(f"Set '{model_name}' as the target model?", default=False):
            self.target = model_name

        # Save configuration
        if self._save_configuration():
            console.print(f"✔ Model '{model_name}' added successfully")
            return 0
        else:
            console.print("❌ Failed to save configuration")
            return 1

    def delete(self) -> int:
        """Interactive method to delete a model configuration."""
        console.print("[bold yellow]Delete Model[/bold yellow]")

        # Check if there are any models to delete
        if not self.models:
            console.print("❌ No models configured to delete")
            return 1

        # List available models
        console.print("Available models:")
        for model_name in self.models.keys():
            is_target = " (active)" if model_name == self.target else ""
            console.print(f"  - {model_name}{is_target}")

        # Get model name to delete
        model_name = Prompt.ask("- Model name to delete")
        if not model_name.strip():
            console.print("❌ Model name cannot be empty")
            return 1

        # Check if model exists
        if model_name not in self.models:
            console.print(f"❌ Model '{model_name}' does not exist")
            return 1

        # Check if it's the target model
        if model_name == self.target:
            console.print(f"⚠️  Warning: '{model_name}' is currently the target model")
            if len(self.models) > 1:
                console.print("You will need to set a new target model after deletion")
            else:
                console.print("❌ Cannot delete the only model. Add another model first.")
                return 1

        # Confirm deletion
        confirm = Confirm.ask(
            f"Are you sure you want to delete model '{model_name}'? This action cannot be undone.",
            default=False,
        )
        if not confirm:
            console.print("❌ Model deletion cancelled")
            return 1

        # Delete model from configuration
        del self.models[model_name]

        # If deleted model was the target, prompt for new target
        if model_name == self.target:
            console.print("\nSelect a new target model:")
            for idx, name in enumerate(self.models.keys(), 1):
                console.print(f"  {idx}. {name}")

            new_target = Prompt.ask("- New target model name", choices=list(self.models.keys()))
            self.target = new_target

        # Save configuration
        if self._save_configuration():
            console.print(f"✔ Model '{model_name}' deleted successfully")
            return 0
        else:
            console.print("❌ Failed to save configuration after deletion")
            return 1

    def set_target(self) -> int:
        """Interactive method to set the target model."""
        console.print("[bold yellow]Set Target Model[/bold yellow]")

        # Check if there are any models
        if not self.models:
            console.print("❌ No models configured. Add a model first.")
            return 1

        # Show current target
        console.print(f"Current target: [bold green]{self.target}[/bold green]\n")

        # List available models
        console.print("Available models:")
        for model_name in self.models.keys():
            is_target = " (current)" if model_name == self.target else ""
            console.print(f"  - {model_name}{is_target}")

        # Get new target model
        new_target = Prompt.ask("- New target model name", choices=list(self.models.keys()), default=self.target)

        if new_target == self.target:
            console.print(f"Target model is already '{new_target}'")
            return 0

        # Update target
        self.target = new_target

        # Save configuration
        if self._save_configuration():
            console.print(f"✔ Target model set to '{new_target}' successfully")
            return 0
        else:
            console.print("❌ Failed to save configuration")
            return 1

    def _save_configuration(self) -> bool:
        """Save configuration to agent.yml file."""
        try:
            # Prepare models section
            models_section = {}
            for model_name, model_config in self.models.items():
                models_section[model_name] = model_config.to_dict()

            # Update configuration
            updates = {
                "models": models_section,
                "target": self.target,
            }
            self.config_manager.update(updates=updates, delete_old_key=True)
            console.print(f"Configuration saved to {self.config_manager.config_path}")
            return True
        except Exception as e:
            console.print(f"❌ Failed to save configuration: {e}")
            logger.error(f"Failed to save configuration: {e}")
            return False
