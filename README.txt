This program implements Q-learning to pick up packages in a four rooms environment.

Scenario1.py deals with picking up a single package.

Scenario2.py deals with picking up 3 packages in any order

Scenario3.py deals with picking up 3 packages in the order red, green, blue.

Stochastic actions are also implemented.

To run the program, run `make` to build the virtual environment, 
then run `source ./venv/bin/activate` to activate the environment,
then run `python3 Scenario{suffix}.py`. 
To run with stochastic actions, add a `-stochastic' flag