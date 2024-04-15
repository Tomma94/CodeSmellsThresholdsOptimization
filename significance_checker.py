import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats


def significance_checker(output_directory, run_number ,results_origin, results_model, plot=True, to_save= True ):

    # Extract the metric names
    metrics = list(results_origin.keys())
    significance_results = {}
    significance_file = f'{output_directory}/{run_number}/significance_results_{run_number}.json'

    # Perform a t-test for each metric
    for metric in metrics:
        t_statistic, p_value = stats.ttest_rel(results_model[metric], results_origin[metric])
        significance_results[metric] = {'t_statistic': t_statistic, 'p_value': p_value, 'value_model': np.mean(results_model[metric]), 'value_origin': np.mean(results_origin[metric])}

    if to_save:
        with open(significance_file, 'w') as json_file:
            json.dump(significance_results, json_file)

    if plot:
        plt.style.use('ggplot')
        # Sample data
        # Sample data
        categories = ['F1 Score', 'AUC-PRC', 'AUC-ROC']
        values_model = [significance_results[i]['value_model'] for i in categories]
        values_origin = [significance_results[i]['value_origin'] for i in categories]
        p_values = [significance_results[i]['p_value'] for i in categories]

        x = np.arange(len(categories))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()

        # Creating bars for the two groups
        bars1 = ax.bar(x - width / 2, values_model, width, label='Generated Thresholds', color='steelblue')
        bars2 = ax.bar(x + width / 2, values_origin, width, label='Origin Thresholds', color='lightskyblue')

        # Adding text for labels, title, and custom x-axis tick labels, etc.
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0, 1)

        # Function to attach a text label above each pair of bars
        def label_diff(i, text, x, Y):
            y = max(Y) + 0.05 * max(Y)  # Adjust vertical position relative to the max value
            ax.text(x[i], y, f"$p-value={text}$", ha='center', va='bottom')  # Use LaTeX formatting

        # Iterating over the bars to annotate the p-value
        for i in range(len(x)):
            # Formatting the p-value for display
            if p_values[i] < 0.001:
                exponent = int(np.floor(np.log10(p_values[i])))
                base = p_values[i] / 10 ** exponent
                p_text = f"{base:.2f} \\times 10^{{{exponent}}}"
            else:
                p_text = f"{p_values[i]:.2f}"
            label_diff(i, p_text, x, [values_model[i], values_origin[i]])

        plt.tight_layout()
        plt.savefig(f'{output_directory}/{run_number}/test_results_{run_number}.png')
        plt.clf()

    return significance_results

