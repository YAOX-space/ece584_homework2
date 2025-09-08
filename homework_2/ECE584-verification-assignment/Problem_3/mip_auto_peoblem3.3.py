import time
import torch
import torch.nn.functional as F
import gurobipy
import argparse
import csv
import os


class SimpleNN(torch.nn.Module):
    """
    Definitition of a simple 3-layer feedforward neural network.
    """

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size_1)
        self.fc2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = torch.nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

class Verifier:

    def __init__(self):
        # Create the MIP solver object.
        self.gurobi_model = gurobipy.Model('MIP')
        # Remove the line below to see more verbose information from Gurobi.
        self.gurobi_model.setParam('OutputFlag', 0)
        self.x_vars = []
        # timeout and logging
        self.run_deadline = None
        self.log_path = os.path.join(os.path.dirname(__file__), 'problem3_log.csv')
        self.method_name = 'MILP'

    def start_run(self, perturbation, timeout_seconds=600):
        """Start a run window and prepare CSV log."""
        self.run_deadline = time.time() + timeout_seconds
        is_new = not os.path.exists(self.log_path)
        if is_new:
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['method', 'perturbation', 'stage', 'objective_idx', 'status', 'obj_value', 'elapsed_s'])
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([self.method_name, perturbation, 'start', '', 'start', '', 0.0])

    def _log(self, perturbation, stage, idx, status, obj_value, elapsed):
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([self.method_name, perturbation, stage, idx, status, obj_value, elapsed])

    def create_inputs(self, x_test, perturbation):
        """
        Create input variables with per-element lower and upper bounds.

        Arguments:
            x_test: a list or 1-dim tensor, containing the input image.
            perturbation: element-wise perturbation added to each element of the input.

        Returns:
            x_vars: A list of input variables with initial upper and lower bounds created.
        """
        for i in range(len(x_test)):
            # TODO : Append to x_vars - a list of variable having lower and upper bounds
            # TODO : Hint - Use .addVar in gurobipy
            # YOUR CODE HERE
            # self.x_vars.append(self.gurobi_model.addVar(...))
            lower_bound = x_test[i] - perturbation
            upper_bound = x_test[i] + perturbation
            self.x_vars.append(self.gurobi_model.addVar(lower_bound, upper_bound))
            
        self.gurobi_model.update()
        print(f"Created {len(self.x_vars)} input variables")
        return self.x_vars

    def add_linear_layer(self, input_vars, weight, bias):
        '''
        LP/MILP formulation for fully connectd layers.

        Inputs:
            input: List of bounds representing input neurons' ranges.
            weight: The weight matrix of the FC layer.
            bias: The bias vector of the FC layer.
            gurobi_model: The current Gurobi linear programming model. Called using self.gurobi_model

        Outputs:
            z: The linear expressions representing the output of this layer before activation.

        Students should calculate the linear expression for each output neuron and determine its bounds.
        '''
        input_size = len(input_vars)
        output_size = len(bias)

        # Pre-activation variables z.
        z = self.gurobi_model.addVars(output_size, lb=float('-inf'), ub=float('inf'))
        self.gurobi_model.update()
        # TODO : Add constraints for variable z representing the output of the layer before activation i.e. wx + b
        input_vars_list = list(input_vars)
        for i in range(output_size):
            # YOUR CODE HERE
            # Add linear constraints of z = w * x + b
            sum_linear = 0
            for j in range(input_size):
                sum_linear += weight[i][j] * input_vars_list[j]
            sum_linear += bias[i]
            self.gurobi_model.addConstr(z[i] == sum_linear)
        self.gurobi_model.update()
        print(f"Created linear layer with input shape {input_size} output shape {output_size}")
        return z.values()

    def add_relu_layer(self, z, lower_bounds, upper_bounds):
        '''
        Adds ReLU activation constraints to the MILP model.

        Inputs:
            z: Linear expressions from the previous FC layer.
            lower_bounds: Lower Bounds for each neuron's output from the previous layer.
            upper_bounds: Upper Bounds for each neuron's output from the previous layer.
            gurobi_model: The current linear programming model. Called using self.gurobi_model

        Outputs:
            hatz_vars: Variables representing the activated outputs, serving as inputs for the next layer.
            gurobi_model: The updated gurobi_model model with ReLU constraints. Called using self.gurobi_model.update()

        Students are tasked with adding constraints to model the ReLU activation within the gurobi_model framework.
        '''
        # Post activation variables.
        n_inputs = len(upper_bounds)
        hatz_vars = self.gurobi_model.addVars(n_inputs)
        # Some p variables will not be used if that neuron is stable.
        p_vars = self.gurobi_model.addVars(n_inputs, vtype=gurobipy.GRB.BINARY)
        self.gurobi_model.update()

        # TODO: Model ReLU activation as piecewise linear constraints in MIP.
        # You need to handle three cases for each neuron based on its pre-activation bounds:
        # active (always positive), inactive (always non-positive), and unstable (can be either).
        # Hint - Use .addConstr() and Refer section 4.1 of the paper
        n_unstable = 0
        for i in range(n_inputs):
            # YOUR CODE HERE
            # Active
            if lower_bounds[i] >= 0:
                self.gurobi_model.addConstr(hatz_vars[i] == z[i])
            # Inactive
            elif upper_bounds[i] <=0:
                self.gurobi_model.addConstr(hatz_vars[i] == 0)
            # Unstabel
            else:
                # counts for unstable + 1
                n_unstable += 1
                
                # constraint 1
                self.gurobi_model.addConstr(hatz_vars[i] >= 0)
                
                # constraint 2
                self.gurobi_model.addConstr(hatz_vars[i] >= z[i])
                
                # constraint 3
                self.gurobi_model.addConstr(hatz_vars[i] <= upper_bounds[i] * p_vars[i])
                
                # constraint 4
                self.gurobi_model.addConstr(hatz_vars[i] <= z[i] - lower_bounds[i] * (1 - p_vars[i]))
                
        self.gurobi_model.update()

        print(f"Created relu layer with {n_unstable} unstable neurons out of {n_inputs} total neurons")
        return hatz_vars.values()

    def solve_objectives(self, objectives, direction='minimization', stage='main', perturbation=None):
        '''
        Solves the Gurobi model for the given objectives.

        Inputs:
            objectives: List of Gurobi linear expressions to be optimized.
            direction: Specifies the optimization direction, either 'minimization' or 'maximization'.
            gurobi_model: The current linear programming model. Called using self.gurobi_model

        Outputs:
            optimal_objs: optimal objective values
            optimal_vars: corresponding input variables at optimality
            timed_out: whether timeout occurred in this stage
        '''
        optimal_objs = []
        optimal_vars = []
        timed_out = False
        for i in range(len(objectives)):
            if objectives[i] is None:
                optimal_objs.append(None)
                optimal_vars.append(None)
                continue
            # early timeout check
            if self.run_deadline is not None and time.time() >= self.run_deadline:
                timed_out = True
                self._log(perturbation, stage, i, 'timeout_early', None, 0.0)
                break
            # set remaining TimeLimit for this objective
            if self.run_deadline is not None:
                remaining = max(1, int(self.run_deadline - time.time()))
                self.gurobi_model.setParam('TimeLimit', remaining)
            else:
                self.gurobi_model.setParam('TimeLimit', 1e9)
            start_time = time.time()
            # TODO : Add Gurobi objectives to the model
            # Implement the objective solving step, where you optimize the verification objectives
            # under the current model constraints. This step is essential for finding potential adversarial examples
            # that satisfy the input constraints and maximize the output difference from the expected label.
            # Hint - direction can either be 'minimzation' or 'maximization' and use self.gurobi_model.setObjective
            if direction == 'minimization':
                # YOUR CODE HERE
                self.gurobi_model.setObjective(objectives[i], gurobipy.GRB.MINIMIZE)
            elif direction == 'maximization':
                # YOUR CODE HERE
                self.gurobi_model.setObjective(objectives[i], gurobipy.GRB.MAXIMIZE)
            else:
                raise ValueError(direction)
            self.gurobi_model.optimize()
            status = self.gurobi_model.status
            time_elapsed = time.time() - start_time
            if status == gurobipy.GRB.OPTIMAL:
                optimal_objs.append(self.gurobi_model.objVal)
                optimal_vars.append([vars.X for vars in self.x_vars])
                self._log(perturbation, stage, i, 'optimal', self.gurobi_model.objVal, time_elapsed)
                print(f'objective {i:3d} {direction} solved in {time_elapsed:8.3f} seconds, obj={self.gurobi_model.objVal:13.8g}')
            elif status == gurobipy.GRB.TIME_LIMIT:
                timed_out = True
                optimal_objs.append(None)
                optimal_vars.append(None)
                self._log(perturbation, stage, i, 'timeout', None, time_elapsed)
                print(f'objective {i:3d} {direction} TIMEOUT after {time_elapsed:8.3f} seconds')
                break
            else:
                optimal_objs.append(None)
                optimal_vars.append(None)
                self._log(perturbation, stage, i, f'status_{status}', None, time_elapsed)
                print(f'objective {i:3d} {direction} ended with status {status}')
        return optimal_objs, optimal_vars, timed_out

    def get_verification_objectives(self, y_vars, groundtruth_label):
        '''
        Generates objectives for verifying the model against adversarial examples.

            Inputs:
                y_vars: Initial input variables ( weights, biases and bounds ).
                groundtruth_label: list of the true class label of the inputs.

            Outputs:
                objectives: list of objectives ( each i contains objective for each input )
                target_labels: list of the class label of the adversarial target.

        '''

        objectives = []
        target_labels = []
        # TODO : Formulate the verification objectives for the MIP model.
        # This involves setting up the optimization targets to assess the neural network's output
        # robustness against input perturbations, particularly focusing on how changes in input
        # can lead to incorrect classifications.
        for i in range(len(y_vars)):
            target_labels.append(i)
            if i == groundtruth_label:
                # YOUR CODE HERE
                objectives.append(None)
                
            # Optimization objective we want to minimize.
            # TODO : append objective that needs to be minimized, think about the original label and the one after perturbation
            # YOUR CODE HERE
            else:
                objectives.append(y_vars[groundtruth_label] - y_vars[i])

        return objectives, target_labels


def load_model_and_data(data):
    # Define model and load the pretrained model weights.
    model = SimpleNN(input_size=28 * 28, hidden_size_1=128, hidden_size_2=64, output_size=10)
    model.load_state_dict(torch.load('model.pth'))

    # find one correct classified image
    model.eval()
    image_sample, sample_label = torch.load(data)
    image_sample = image_sample.reshape(1, -1)
    outputs = model(image_sample)
    predicted_label = outputs.argmax(dim=1)

    return model, image_sample[0], sample_label


def extract_model_weight(model):
    weights = []
    biases = []
    i = 1
    state_dict = model.state_dict()
    while f'fc{i}.weight' in state_dict:
        weights.append(state_dict[f'fc{i}.weight'.format(i)])
        biases.append(state_dict[f'fc{i}.bias'.format(i)])
        i += 1
    return weights, biases


def verify(model, input_image, groundtruth_label, perturbation):
    print(f"Verifiying Pertubation - {perturbation}")
    v = Verifier()
    v.start_run(perturbation, timeout_seconds=600)
    # Create input variables.
    variables = v.create_inputs(input_image, perturbation)
    # Extract model weights and biases.
    weights, biases = extract_model_weight(model)
    num_layers = len(weights)
    # formulation for non-final layer.
    for i in range(num_layers):
        variables = list(v.add_linear_layer(variables, weights[i], biases[i]))
        if i != num_layers - 1:
            # For all linear layers except for the last layer, calculate intermediate layer bounds.
            print(f'Calculating intermediate layer bounds for {len(variables)} neurons...')
            lbs, _, to1 = v.solve_objectives(variables, direction='minimization', stage=f'layer{i}_lb', perturbation=perturbation)
            if to1:
                print('timeout')
                return False
            ubs, _, to2 = v.solve_objectives(variables, direction='maximization', stage=f'layer{i}_ub', perturbation=perturbation)
            if to2:
                print('timeout')
                return False
            # After obtaining intermediate layer bounds, we can formulate the ReLU layer.
            variables = v.add_relu_layer(variables, lbs, ubs)

    # Create verification objectives.
    objectives, target_labels = v.get_verification_objectives(variables, groundtruth_label)
    print(f'Optimizing verification objectives...')
    optimal_objs, optimal_solutions, to3 = v.solve_objectives(objectives, direction='minimization', stage='final', perturbation=perturbation)
    if to3:
        print('timeout')
        return False

    verified = True
    for (i, r) in enumerate(target_labels):
        if r == groundtruth_label:
            continue
        model_pred = model(torch.tensor(optimal_solutions[i]))
        margin = model_pred[groundtruth_label] - model_pred[r]
        # For LP, below we print a lower bound of the objective rather than the true minimum.
        print(f'Minimum y_{groundtruth_label} - y{r} = {optimal_objs[i]:13.8g}; calculated margin = {margin:13.8g}')
        # The assertion is for MIP only. For LP, this assertion should be removed.
        assert abs(margin - optimal_objs[i]) < 1e-2
        if optimal_objs[i] <= 0:
            verified = False
            print(
                f'A counterexample found - with perturbation {perturbation} the label {r} has a score greater than groundtruth')

    return verified


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add the 'data' argument
    parser.add_argument('data_file', type=str, help='model input x_test loaded from file (e.g., data1.pth)')
    # Parse the command line arguments
    args = parser.parse_args()
    model, x_test, groundtruth_label = load_model_and_data(args.data_file)
    start_time = time.time()
    perturbation_sizes = [0, 0.001, 0.003, 0.01, 0.03, 0.1]
    for s in perturbation_sizes:
        result = verify(model, x_test, groundtruth_label, perturbation=s)
        verification_time = time.time() - start_time
        print(f'Verification result: {result} in {verification_time} seconds')
    