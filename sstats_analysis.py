"""
A metric system for measuring the fit of proposed params
via summary statistics
Needs work.

Author: Sara Mathieson, Rebecca Riley
Date 06/28/2023
"""
import numpy as np
import optparse

from math import sqrt
from ss_helpers import parse_output
from ss_helpers import stats_all
from util import process_opts

NUM_TRIAL = 100
NUM_BINS = 10

def main():
    # setup --------------------------------------------------------------------
    opts, param_values = parse_args() # also handles infile reading
    generator, iterator, parameters, sample_sizes = process_opts(opts,
        summary_stats=True)
    generator.update_params(param_values)
    print("VALUES", param_values)
    print("made it through params")

    # get statistics -----------------------------------------------------------

    '''
    NOTE: for summary stats, use neg1=False to keep hap data as 0/1 (not -1/1)
    NOTE: use region_len=True for Tajima's D (i.e. not all regions have same S)
    '''
    # real --> Truth
    real_matrices = iterator.real_batch(batch_size=NUM_TRIAL, neg1=False)
    real_matrices_region = iterator.real_batch(batch_size=NUM_TRIAL, neg1=False,
        region_len=True)

    # sim --> Experiment
    sim_matrices = generator.simulate_batch(batch_size=NUM_TRIAL, neg1=False)
    sim_matrices_region = generator.simulate_batch(batch_size=NUM_TRIAL,
        neg1=False, region_len=True)

    num_pop = len(sample_sizes)
    assert num_pop == 1 # TODO multipop

    truth_values_stats = stats_all(real_matrices, real_matrices_region)
    exp_values_stats = stats_all(sim_matrices, sim_matrices_region)

    # difference calculation -------------------------------------------------
    # pop_sfs, pop_dist, pop_ld, Tajima's D, pi, and num_haplotypes

    num_stats = len(truth_values_stats)
    stat_error = np.zeros((num_stats))

    for i in range(num_stats):
        truth_values = truth_values_stats[i]
        exp_values = exp_values_stats[i]

        stat_error[i] = calculate_binned_error(truth_values, exp_values, i)

    print("\n\n")
    result = ""
    for i in range(num_stats):
        # print(stat_error[i])
        result = result + str(stat_error[i]) + "\t"
    print(result)

    complete_error = np.sum(stat_error) / num_stats
    print(complete_error)

def error2(truth, exp):
    if truth == 0.:
        return 0.

    return (truth - exp) ** 2 / truth ** 2

def binned_error(n, truth, exp):
    bin_size = n // NUM_BINS
    truth_values_sorted = np.sort(truth)
    bin_maxes = [float('inf') for j in range(NUM_BINS)]

    for j in range(NUM_BINS-1):
        bin_maxes[j] = truth_values_sorted[(j+1)*bin_size]

    truth_bin_counts = [0 for k in range(NUM_BINS)]
    bin_pointer = 0

    for j in range(n):
        while truth_values_sorted[j] >= bin_maxes[bin_pointer]:
            bin_pointer += 1

        truth_bin_counts[bin_pointer] += 1

    exp_bin_counts = [0 for k in range(NUM_BINS)]
    bin_pointer = 0
    exp_values_sorted = np.sort(exp)

    for j in range(n):
        while exp_values_sorted[j] >= bin_maxes[bin_pointer]:
            bin_pointer += 1

        exp_bin_counts[bin_pointer] += 1

    error_sum = 0.
    for j in range(NUM_BINS):
        error_sum += error2(truth_bin_counts[j], exp_bin_counts[j])
    
    return error_sum / NUM_BINS

def calculate_binned_error(truth_values, exp_values, i):
    N = len(truth_values)
    assert N == len(exp_values)

    if i == 0 or i == 2:
        total_error = 0.
        for j in range(N):
            total_error += binned_error(len(truth_values), 
                truth_values[j], exp_values[j])
        return total_error / N

    # else
    return binned_error(N, truth_values, exp_values)
    

def calculate_mse(truth_values, exp_values, i):
    def mse(truth, exp):
        return (truth - exp) ** 2

    stat_error_value = 0.

    for j in range(len(truth_values)):
        if i == 0 or i == 2: # these are multi-layered
            truth_values2 = np.sort(truth_values[j])
            exp_values2 = np.sort(exp_values[j])
            for k in range(len(truth_values2)):
                stat_error_value += mse(truth_values2[k], exp_values2[k])
        else:
            truth_values2 = np.sort(truth_values)
            exp_values2 = np.sort(exp_values)
            stat_error_value += mse(truth_values2[j], exp_values2[j])

    return stat_error_value

def parse_args():
    parser = optparse.OptionParser(description='parsing sstat analysis args')

    parser.add_option('-i', '--infile', type='string',help='trial data file',
        default=None)
    
    parser.add_option('-m', '--model', type='string',help='exp, im, ooa2, ooa3')
    parser.add_option('-p', '--params', type='string',
        help='comma separated parameter list')
    parser.add_option('-d', '--data_h5', type='string', help='real data file')
    parser.add_option('-b', '--bed', type='string', help='bed file (mask)')
    parser.add_option('-r', '--reco_folder', type='string',
        help='recombination maps')
    parser.add_option('-n', '--sample_sizes', type='string',
        help='comma separated sample sizes for each population, in haps')
    parser.add_option('-s', '--seed', type='int', default=1833,
        help='seed for RNG')

    parser.add_option('-v', '--param_values', type='string',
        help='comma separated values corresponding to params', default=None)

    (opts, args) = parser.parse_args()

    if opts.infile is not None:
        param_values_infile, in_file_data = parse_output(opts.infile)

        if opts.model is None:
            opts.model = in_file_data['model']

        if opts.params is None:
            opts.params = in_file_data['params']

        if opts.data_h5 is None:
            opts.data_h5 = in_file_data['data_h5']

        if opts.bed is None:
            opts.bed = in_file_data['bed_file']

        if opts.sample_sizes is None:
            opts.sample_sizes = in_file_data['sample_sizes']

        if opts.reco_folder is None:
            opts.reco_folder = in_file_data['reco_folder']

    if opts.param_values is None:
        param_values = param_values_infile
    else:
        arg_values = [float(val_str) for val_str in
            opts.param_values.split(',')]
        param_values = arg_values

    mandatories = ['model','params','sample_sizes']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts, param_values

if __name__ == "__main__":
    main()
