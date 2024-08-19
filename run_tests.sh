#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")"

# Set the PYTHONPATH to the current directory (project root)
export PYTHONPATH=$(pwd)

# Check if pytest is installed
if command -v pytest &> /dev/null
then
    echo "pytest found, running tests with pytest..."
    pytest -s tests/
else
    echo "pytest not found, running tests with unittest..."
    python -m unittest -b -v discover -s tests
fi

# Optional: add other tasks to automate here

echo "Tests completed."
