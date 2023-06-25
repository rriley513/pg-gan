"""
Author: Sara Mathieson, Rebecca Riley
Date 06/22/2023
"""
import optparse

from ss_helpers import stats_all
import util

NUM_TRIAL = 2

def main():
    # setup --------------------------------------------------------------------
    generator, iterator, parameters, sample_sizes, param_values = setup()
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


    # pop_sfs
    sfs_error = 0

    truth_sfs = real_stats_pop[0]
    exp_sfs = sim_stats_pop[0]
    num_sfs = len(truth_sfs)

    for i in range(num_sfs):
        for j in range(len(truth_sfs[i])):
            sfs_error += mse(truth_sfs[i][j], exp_sfs[i][j])

    # pop_dist
    truth_pop_dist = real_stats_pop[1]
    exp_pop_dist = sim_stats_pop[1]

    print(sfs_error)
    

def setup():
    opts = parse_args()
    print("parsed args")

    # load args
    if opts.infile:
        param_values, in_file_data = ss_helpers.parse_output(input_file)
    else:
        param_values = opts.param_values
        in_file_data = None

    opts, param_values = util.parse_args(in_file_data = in_file_data,
        param_values=param_values)

    print("parsing complete")

    generator, iterator, parameters, sample_sizes = util.process_opts(opts,
        summary_stats=True)

    print("generator etc loaded")
    return generator, iterator, parameters, sample_sizes, param_values

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

    parser.add_option('-v', '--param_values', type='string',
        help='comma separated values corresponding to params', default=None)

    (opts, args) = parser.parse_args()
    return opts

if __name__ == "__main__":
    main()