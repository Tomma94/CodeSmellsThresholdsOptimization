import torch
import torch.nn as nn

class DynamicNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        num_layers = num_layers-1
        super(DynamicNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        if num_layers > 0 :
            self.layers += [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        x = torch.sigmoid(x)
        return x

def calc_smells(thresholds, X):
    '''
    This function gets the thresholds and the instances and returns for each instance the smells features based on the thresholds and the metrics.
    '''

    # Define smells as a tensor
    smells = torch.tensor([[[0, 0, 1], [-1, -1, -1], [-1, -1, -1]], [[1, 1, 1], [-1, -1, -1], [-1, -1, -1]],
                           [[0, 2, 0], [-1, -1, -1], [-1, -1, -1]], [[0, 3, 1], [2, 4, 0], [-1, -1, -1]],
                           [[3, 5, 1], [4, 6, 1], [5, 7, 1]], [[6, 8, 1], [7, 9, 0], [-1, -1, -1]],
                           [[8, 10, 1], [-1, -1, -1], [-1, -1, -1]], [[9, 0, 2], [10, 11, 1], [-1, -1, -1]],
                           [[11, 0, 2], [6, 12, 1], [-1, -1, -1]], [[11, 0, 2], [6, 13, 0], [-1, -1, -1]],
                           [[12, 14, 1], [-1, -1, -1], [-1, -1, -1]]], dtype=torch.float64)
    #                             large class                    refused_bequest                 lazy_class                      god_class                 multifaceted_abstraction    hublike_modularization             deep_hierarchy                 swiss_army_knife                  unnecessary_abstraction        broken_modularization        class_data_should_be_private

    # Get the shapes of X and smells
    num_instances = X.shape[0]
    num_smells = smells.shape[0]
    max_conditions = smells.shape[1]

    # Create a new tensor to store the results
    results = torch.zeros((num_instances, num_smells), dtype=torch.bool)

    # Iterate over each instance in the "X" tensor
    for instance_idx in range(num_instances):
        metrics = X[instance_idx]
        threshold = thresholds[instance_idx]

        # Iterate over all the smells
        for s_idx in range(num_smells):
            s_values = smells[s_idx]

            # Initialize a tensor to keep track of the AND result for this smell
            instance_and_result = torch.ones(max_conditions, dtype=torch.bool)

            # Iterate over the conditions for this smell
            for cond in range(max_conditions):
                cond_metric_idx = int(s_values[cond, 0])
                cond_threshold_idx = int(s_values[cond, 1])
                cond_is_greater = int(s_values[cond, 2])

                # Skip if any value in the instance is -1
                if cond_metric_idx == -1:
                    break

                cond_metric_value = metrics[cond_metric_idx].float()
                cond_threshold_value = threshold[cond_threshold_idx].float()
                if cond_is_greater == 1:
                    instance_and_result &= (cond_metric_value > cond_threshold_value)
                elif cond_is_greater == 2:
                    instance_and_result &= (cond_metric_value == 1)
                else:
                    instance_and_result &= (cond_metric_value < cond_threshold_value)

            # Assign the final AND result for this instance to the "results" tensor
            results[instance_idx, s_idx] = instance_and_result.all()

    # Convert the boolean tensor to an integer tensor
    results = results.int()

    return results.to(torch.float32)