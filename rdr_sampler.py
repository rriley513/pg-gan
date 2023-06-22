'''
A real data random iterator with an extremely small sample data set.
For easier visualization and introduction to the real data 
random iterator and its methods.

Not recommended for or tested on serious use (pg_gan, summary_stats)

Author: Rebecca Riley
Date: 06/22/2023
'''
import numpy as np
from numpy.random import default_rng

from global_vars import DEFAULT_SEED
from global_vars import parse_chrom
from real_data_random import Region
from real_data_random import read_mask
from util import major_minor

SAMPLE_L = 20
SAMPLE_NUM_SNPS = 2
SAMPLE_BATCH_SIZE = 2
SEED = DEFAULT_SEED

class RDRSampler:
    def __init__(self, chrom_starts=False):
        # sample has 5 individuals, 11 SNPs, and only uses chroms 1, 2, and 3
        self.haps_all = np.array([[1,0,0,1,0], [0,0,0,0,1], [1,0,1,1,0], 
            [1,0,1,0,0], [0,1,0,0,1], [0,0,1,0,0], [1,0,0,1,0], 
            [0,1,1,1,1], [1,0,0,0,1], [0,1,1,0,0], [1,0,1,0,1]])
        self.pos_all = np.array([12,20,30,33,40,10,30,44,18,21,60])
        self.chrom_all = np.array([1,1,1,1,1,2,2,2,3,3,3])

        print("after haps", self.haps_all.shape)
        self.num_samples = self.haps_all.shape[1]
        self.num_snps = len(self.pos_all)

        self.mask_dict = read_mask("sample.bed")

        self.rng = default_rng(SEED) # reproducible randomness

        if chrom_starts:
            self.chrom_counts = None

    # copies of rdr iterator methods with smaller variable sizes
    def find_end(self, start_idx):
        """
        Based on the given start_idx and the region_len, find the end index
        """
        ln = 0
        chrom = parse_chrom(self.chrom_all[start_idx])
        i = start_idx
        curr_pos = self.pos_all[start_idx]
        while ln < SAMPLE_L:

            if len(self.pos_all) <= i+1:
                print("not enough on chrom", chrom)
                return -1 # not enough on last chrom

            next_pos = self.pos_all[i+1]
            if parse_chrom(self.chrom_all[i+1]) == chrom:
                diff = next_pos - curr_pos
                ln += diff
            else:
                print("not enough on chrom", chrom)
                return -1 # not enough on this chrom
            i += 1
            curr_pos = next_pos

        return i # exclusive

    def real_region(self, neg1, region_len):
        # inclusive
        start_idx = self.rng.integers(0, self.num_snps - SAMPLE_NUM_SNPS)

        if region_len:
            end_idx = self.find_end(start_idx)
            if end_idx == -1:
                return self.real_region(neg1, region_len) # try again
        else:
            end_idx = start_idx + SAMPLE_NUM_SNPS # exclusive

        # make sure we don't span two chroms
        start_chrom = self.chrom_all[start_idx]
        end_chrom = self.chrom_all[end_idx-1] # inclusive here

        if start_chrom != end_chrom:
            #print("bad chrom", start_chrom, end_chrom)
            return self.real_region(neg1, region_len) # try again

        hap_data = self.haps_all[start_idx:end_idx, :]
        start_base = self.pos_all[start_idx]
        end_base = self.pos_all[end_idx]
        positions = self.pos_all[start_idx:end_idx]

        chrom = parse_chrom(start_chrom)
        region = Region(chrom, start_base, end_base)
        result = region.inside_mask(self.mask_dict)

        # if we do have an accessible region
        if result:
            # if region_len, then positions_S is actually positions_len
            dist_vec = [0] + [(positions[j+1] - positions[j])/SAMPLE_L
                for j in range(len(positions)-1)]

            after = process_gt_dist(hap_data, dist_vec,
                region_len=region_len, real=True, neg1=neg1)
            return after

        # try again if not in accessible region
        return self.real_region(neg1, region_len)

    def real_batch(self, batch_size = SAMPLE_BATCH_SIZE, neg1=True,
        region_len=False):
        """Use region_len=True for fixed region length, not by SNPs"""

        if not region_len:
            regions = np.zeros((batch_size, self.num_samples,
                SAMPLE_NUM_SNPS, 2), dtype=np.float32)

            for i in range(batch_size):
                regions[i] = self.real_region(neg1, region_len)

        else:
            regions = []
            for i in range(batch_size):
                regions.append(self.real_region(neg1, region_len))

        return regions

# have to copy in this function to swap out num_snps
def process_gt_dist(gt_matrix, dist_vec, region_len=False, real=False,
    neg1=True):
    """
    Take in a genotype matrix and vector of inter-SNP distances. Return a 3D
    numpy array of the given n (haps) and S (SNPs) and 2 channels.
    Filter singletons at given rate if filter=True
    """
    num_SNPs = gt_matrix.shape[0] # SNPs x n
    n = gt_matrix.shape[1]

    # double check
    if num_SNPs != len(dist_vec):
        print("gt", num_SNPs, "dist", len(dist_vec))
    assert num_SNPs == len(dist_vec)

    # used for trimming (don't trim if using the entire region)
    S = num_SNPs if region_len else SAMPLE_NUM_SNPS

    # set up region
    region = np.zeros((n, S, 2), dtype=np.float32)

    mid = num_SNPs//2
    half_S = S//2
    if S % 2 == 1: # odd
        other_half_S = half_S+1
    else:
        other_half_S = half_S

    # enough SNPs, take middle portion
    if mid >= half_S:
        minor = major_minor(gt_matrix[mid-half_S:mid+
            other_half_S,:].transpose(), neg1)
        region[:,:,0] = minor
        distances = np.vstack([np.copy(dist_vec[mid-half_S:mid+other_half_S])
            for k in range(n)])
        region[:,:,1] = distances

    # not enough SNPs, need to center-pad
    else:
        print("NOT ENOUGH SNPS", num_SNPs)
        print(num_SNPs, S, mid, half_S)
        minor = major_minor(gt_matrix.transpose(), neg1)
        region[:,half_S-mid:half_S-mid+num_SNPs,0] = minor
        distances = np.vstack([np.copy(dist_vec) for k in range(n)])
        region[:,half_S-mid:half_S-mid+num_SNPs,1] = distances

    return region # n X SNPs X 2

if __name__ == "__main__":
    RDRSampler().real_batch(5)
