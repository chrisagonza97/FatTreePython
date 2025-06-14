A simulation and optimization framework for virtual machine (VM) placement and migration in Fat-Tree data center topologies. Includes implementations of placement algorithms and cost evaluation using PyTorch-based methods.

---

## Project Structure

    FatTreePython/
    ├── src/              # Core logic and simulation engine
    │   └── app.py        # Main entry point
    ├── test/             # Unit tests
    │   └── test_app.py   # Test cases for core components
    ├── README.md

---

## Requirements

Install dependencies (if using `requirements.txt`):

## How to Run the Application (Not currently complete)

Run the main FatTree simulation from the root directory:

    python -m src.app

This will execute the main logic inside `src/app.py`, initializing the FatTree structure, placing VMs, and running cost-based simulations.

---

## How to Run Tests

To execute unit tests in `test/test_app.py`:

    python -m test.test_app

Example output:

    ......
    ----------------------------------------------------------------------
    Ran 6 tests in 0.012s

    OK

**Note:** Make sure `test/` contains an `__init__.py` file so Python treats it as a package.

---

## Notes

- `src/` and `test/` are structured as Python packages.
- To run modules with `-m`, always execute from the project root.
