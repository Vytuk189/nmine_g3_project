

## Current Status

The current configuration in `main.py` solves **Task 15 using the explicit Euler method**. It produces the expected results for this task, so I'm guessing the implementation is correct.

I haven't yet started writing the documentation and function comment, so consult a chatbot for explanation if necessary.

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Configures and runs the computation. |
| `domain_rules.py` | Defines the node-density and cross-sectional-area rules. |
| `conditions.py` | Defines initial conditions and fixed boundary conditions. |
| `data_io.py` | Handles saving and loading computation results. |

Other scripts are focused on building the solver.

## Configuring a Computation

The intended workflow is to configure the computation in `main.py` and then run that script.

- Define or modify node-density and cross-sectional-area rules in `domain_rules.py`.
- Define or modify initial conditions and fixed boundary conditions in `conditions.py`.
- Select the desired rules and computation settings in `main.py`.

The long-term goal is to build a collection of reusable domain rules, initial conditions, and fixed boundary conditions. Users will then be able to configure different problems by selecting existing rules in `main.py`.

## Running the Code

Open a terminal in the project folder and run:

```bash
python main.py

