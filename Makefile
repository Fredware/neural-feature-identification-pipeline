# Get the Conda env name from environment.yaml
ENV_NAME := $(shell grep 'name:' environment.yaml | cut -d ' ' -f 2)
SHELL_INIT = module purge && module use $(HOME)/mymodules && module load pycharm miniforge3

# Declare targets that are not files (i.e., they are command labels). These are actions to be performed not target files to be created
.PHONY: help install setup-conda install-python install-r pipeline clean lint format test

# Set the default command to 'help' when 'make' is run without arguments
.DEFAULT_GOAL := help

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: setup-conda install-python install-r ## Run the complete one-time project setup (Conda env + uv Python package install + Rscript package install)

setup-conda: ## Create Conda environment from environment.yaml
	@echo "---Creating Conda environment: $(ENV_NAME) ---"
	@bash -c "$(SHELL_INIT) && conda env create -f environment.yaml"

install-python: ## Install Python packages into the Conda environment via uv
	@echo "--- Installing Python packages via uv ---"
	bash -c "$(SHELL_INIT) && conda run -n $(ENV_NAME) uv pip install -e '.[dev]'"

install-r: ## Install R packages via Rscript
	@echo "--- Installing R packages via Rscript ---"
	bash -c "$(SHELL_INIT) && conda run -n $(ENV_NAME) Rscript scripts/r/install_r_packages.R"

pipeline: ## Run the main Snakemake pipeline
	@echo "--- Running the Snakemake pipeline (ensure env is active) ---"
	snakemake --cores all --use-conda

clean: ## Remove Python cache files
	@echo "--- Removing Conda environment: $(ENV_NAME) ---"
	bash -c "$(SHELL_INIT) && conda env remove -n $(ENV_NAME)"
	@echo "--- Removing Python cache files ---"
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint: ## Check code formatting and style
	@echo "--- Linting code with ruff ---"
	conda run -n $(ENV_NAME) ruff check .

format: ## Automatically format code
	@echo "--- Linting code with ruff ---"
	conda run -n $(ENV_NAME) ruff format .

test: ## Run tests using pytest
	@echo "--- Running tests ---"
	conda run -n $(ENV_NAME) pytest
