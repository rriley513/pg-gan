"""
Author: Sara Mathieson, Rebecca Riley
Date 06/22/2023
"""
import numpy as np
import optparse

from math import sqrt
from ss_helpers import parse_output
from ss_helpers import stats_all
from util import process_opts

NUM_TRIAL = 100

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

    real_stats_pop = stats_all(real_matrices, real_matrices_region)
    sim_stats_pop = stats_all(sim_matrices, sim_matrices_region)

    # difference calculation -------------------------------------------------
    # pop_sfs, pop_dist, pop_ld, Tajima's D, pi, and num_haplotypes
    def mse(truth, exp):
        return (truth - exp) ** 2

    num_stats = len(real_stats_pop)
    stat_error = np.zeros((num_stats))

    for i in range(num_stats):
        truth_values = real_stats_pop[i]
        exp_values = sim_stats_pop[i]
        
        for j in range(len(truth_values)):
            if i == 0 or i == 2: # these are multi-layered
                truth_values2 = np.sort(truth_values[j])
                exp_values2 = np.sort(exp_values[j])
                for k in range(len(truth_values2)):
                    stat_error[i] += mse(truth_values2[k], exp_values2[k])
            else:
                truth_values2 = np.sort(truth_values)
                exp_values2 = np.sort(exp_values)
                stat_error[i] += mse(truth_values2[j], exp_values2[j])

    result = ""
    for i in range(num_stats):
        # print(stat_error[i])
        result = result + str(stat_error[i] / NUM_TRIAL) + "\t"
    print(result)

    complete_error = np.sum(stat_error)
    print(complete_error)

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
