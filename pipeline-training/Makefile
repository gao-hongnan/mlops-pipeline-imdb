SHELL = /bin/bash

.PHONY: help venv
help:
	@echo "Commands:"
	@echo "venv VENV_NAME=<venv_name> PYPROJECT=<pyproject> DEV=<dev> : creates a virtual environment with optional development dependencies."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."
	@echo "test    : execute tests on code, data and models."
	@echo "create_repo_template : create repository template with necessary directories and files."

# setup virtual environment
# make venv VENV_NAME=venv DEV=dev
.PHONY: venv
venv:
	@curl -o scripts/make_venv.sh \
	https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh
	@chmod +x scripts/make_venv.sh
	@scripts/setup_virtual_env.sh $(VENV_NAME) $(PYPROJECT) $(DEV)

# Create Repo Template
.PHONY: create_repo_template
create_repo_template:
	@chmod +x scripts/create_repo_template.sh
	@scripts/create_repo_template.sh

# Styling
.PHONY: style
style:
	@chmod +x scripts/formatter.sh
	@scripts/formatter.sh
	@chmod +x scripts/linter.sh
	@scripts/linter.sh

# Cleaning
.PHONY: clean
clean:
	@chmod +x scripts/clean.sh
	@scripts/clean.sh

# Tests
.PHONY: test
test:
	@chmod +x scripts/run_tests.sh
	@scripts/run_tests.sh