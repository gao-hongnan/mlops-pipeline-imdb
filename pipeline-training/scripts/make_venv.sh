#!/bin/bash

# curl -o make_venv.sh \
#   https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Successfully fetched and sourced the 'utils.sh' script from the 'common-utils' repository on GitHub."

usage() {
    echo "Usage: $0 <venv_name> [--pyproject] [--dev]"
    echo
    echo "Creates a virtual environment and installs dependencies."
    echo
    echo "Arguments:"
    echo "  venv_name                The name of the virtual environment to create."
    echo "  --pyproject              Install dependencies from pyproject.toml instead of requirements files."
    echo "  --dev                    Install dev dependencies (only applicable with --pyproject)."
    echo
    exit 1
}

check_input() {
    if [ -z "$1" ]; then
        echo "Error: Virtual environment name not provided."
        usage
        exit 1
    fi
}

create_venv() {
  local venv_name="$1"
  python3 -m venv "$venv_name"
}

activate_venv() {
  local venv_name="$1"
  source "$venv_name/bin/activate" || source "$venv_name/Scripts/activate"
}

upgrade_pip() {
  python3 -m pip install --upgrade pip setuptools wheel
}

install_dependencies() {
  local pyproject="$1"
  local dev="$2"

  if [ "$pyproject" = "--pyproject" ]; then
    logger "INFO" "Installing dependencies from pyproject.toml"
    if [ "$dev" = "--dev" ]; then
      logger "INFO" "Installing dev dependencies as well..."
      python3 -m pip install -e ".[dev]"
    else
      python3 -m pip install -e .
    fi
  else
    logger "INFO" "Installing dependencies from requirements files..."
    logger "INFO" "Please name your requirements files requirements.txt and requirements_dev.txt."
    local requirements_path="requirements.txt"
    local requirements_dev_path="requirements_dev.txt"
    if [ -f "$requirements_path" ]; then
      python3 -m pip install -r "$requirements_path"
    fi
    if [ -f "$requirements_dev_path" ]; then
      logger "INFO" "Installing dev requirements"
      python3 -m pip install -r "$requirements_dev_path"
    fi
  fi
}

main() {
  if [ $# -lt 1 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
  fi

  local venv_name="$1"
  local pyproject=""
  local dev=""

  for arg in "$@"
  do
    case $arg in
        --pyproject)
        pyproject="--pyproject"
        shift
        ;;
        --dev)
        dev="--dev"
        shift
        ;;
    esac
  done

  check_input "$venv_name"

  create_venv "$venv_name"
  activate_venv "$venv_name"
  upgrade_pip
  install_dependencies "$pyproject" "$dev"
}

main "$@"
