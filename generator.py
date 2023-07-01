"""
Generator class for pg-gan.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
from numpy.random import default_rng

# our imports
import global_vars
from param_set import ParamSet
import simulation
import util

################################################################################
# GENERATOR CLASS
################################################################################

class Generator:

    def __init__(self, simulator, iterable_params, sample_sizes, seed,
        mirror_real=False, reco_folder=""):
        self.simulator = simulator
        self.iterable_params = iterable_params
        self.sample_sizes = sample_sizes
        self.num_samples = sum(sample_sizes)
        self.rng = default_rng(seed)

        # for real data, use HapMap
        if mirror_real and reco_folder != None:
            files = global_vars.get_reco_files(reco_folder)

            self.prior, self.weights = util.parse_hapmap_empirical_prior(files)

        else:
            self.prior, self.weights = [], []

    def simulate_batch(self, batch_size=global_vars.BATCH_SIZE, params=[], 
        region_len=False, real=False, neg1=True, mutation_rates=None):
        snp_rates = np.zeros((batch_size, 1))

        # initialize 4D matrix (two channels for distances)
        if region_len:
            regions = []
        else:
            regions = np.zeros((batch_size, self.num_samples,
                global_vars.NUM_SNPS, 2), dtype=np.float32) # two channels

        # set up parameters
        sim_params = ParamSet(self.simulator)
        if real:
            pass # keep orig for "fake" real
        elif params == []:
            sim_params.update(self.iterable_params)
        else:
            sim_params.update(params)

        default_mut = sim_params.get("mut")

        # simulate each region
        for i in range(batch_size):
            if mutation_rates is not None:
                region, num_snps = self.mut_est_region(sim_params, mutation_rates[i], 
                    neg1, region_len, default_mut)
            else:
                seed = self.rng.integers(1,high=2**32) # like GAN "noise"

                ts = self.simulator(sim_params, self.sample_sizes, seed,
                    self.get_reco(sim_params.get("reco")))
                num_snps = ts.genotype_matrix().astype(float).shape[0]
                region, _ = prep_region(ts, neg1, region_len=region_len)

            if region_len:
                regions.append(region)
            else:
                regions[i] = region
            snp_rates[i] = num_snps/global_vars.L

        return regions, snp_rates

    def mut_est_region(self, sim_params, proposed_mut, neg1, region_len, default_mut):
        if proposed_mut > 0: # sometimes due to XL regions, get negative estimated mut_rate
            mut_rate = proposed_mut
        else:
            mut_rate = default_mut
        sim_params.param_set["mut"].value = mut_rate

        sufficient_snps = False
        tries = 0
        while not sufficient_snps and tries < 10:
            seed = self.rng.integers(1,high=2**32) # like GAN "noise"
            tries += 1
            ts = self.simulator(sim_params, self.sample_sizes, seed, \
                self.get_reco(sim_params.get("reco")))
            gt_matrix = ts.genotype_matrix().astype(float)
            num_snps = gt_matrix.shape[0]

            if num_snps > 0:
                region, sufficient_snps = prep_region(ts, neg1, region_len)

        if not sufficient_snps and tries == 10:
            # let assertion error happen at this point. It shouldn't be that common.

            print("WARNING: NOT ENOUGH SNPS, USING DEFAULT MUTATION RATE (" + \
                  str(sim_params.mut.value) + ")")
            sim_params.mut.value = default_mut
            ts = self.simulator(sim_params, self.sample_sizes, seed, \
                reco=self.get_reco(sim_params.get("reco")))
            gt_matrix = ts.genotype_matrix().astype(float)
            num_snps = gt_matrix.shape[0]
            assert num_snps > 0
            region, _ = prep_region(gt_matrix, ts.variants(), neg1, region_len)

        return region, num_snps

    def real_batch(self, batch_size = global_vars.BATCH_SIZE, neg1=True,
        region_len=False):
        return self.simulate_batch(batch_size=batch_size, real=True, neg1=neg1,
            region_len=region_len)

    def update_params(self, new_params):
        self.iterable_params = new_params

    def update_param_values(self, param_names, param_values):
        if not self.iterable_params:
            self.iterable_params = ParamSet(self.simulation)

        num_params = len(param_names)
        assert num_params == len(param_values)

        for i in range(num_params):
            param_name = param_names[i]

            if param_name in self.iterable_params.param_set:
                self.iterable_params.param_set[param_name].value\
                    = param_values[i]

    def get_reco(self, reco):
        if self.prior == []:
            return reco

        return draw_background_rate_from_prior(self.prior, self.weights,
            self.rng)

def draw_background_rate_from_prior(prior_rates, prob, rng):
    return rng.choice(prior_rates, p=prob)

def prep_region(ts, neg1, region_len):
    """Gets simulated data ready"""
    gt_matrix = ts.genotype_matrix().astype(float)
    snps_total = gt_matrix.shape[0]

    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total
    dist_vec = [0] + [(positions[j+1] - positions[j])/global_vars.L for j in
        range(snps_total-1)]

    # when mirroring real data
    return util.process_gt_dist(gt_matrix, dist_vec, region_len=region_len,
        neg1=neg1)

# testing
if __name__ == "__main__":
  
    batch_size = 50
    simulator = simulation.exp
    param_names = ["N1", "T1"]
    params = ParamSet(simulator, param_names)

    # quick test
    print("sim exp")
    generator = Generator(simulator, params, [20],
                          global_vars.DEFAULT_SEED)
    generator.update_params(params)
    mini_batch = generator.simulate_batch(batch_size=batch_size)
    print("x", mini_batch[0,:,:,0], mini_batch.shape)
