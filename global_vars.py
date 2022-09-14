'''For collecting global values'''
# section A: general -----------------------------------------------------------
NUM_SNPS = 36       # number of seg sites, should be divisible by 4
L = 50000           # heuristic to get enough SNPs for simulations (50,000 or fifty-thousand)
CHROM_RANGE = range(1,23)
BATCH_SIZE = 50

DEFAULT_SEED = 1833
DEFAULT_SAMPLE_SIZE = 198

FRAC_TEST = 0.1 # depricated

# section B: overwriting in-file data-------------------------------------------

# to use custom trial data, switch OVERWRITE_TRIAL_DATA to True and
# change the TRIAL_DATA dictionary to have the values desired.
# Model, params, and param_values must be defined
OVERWRITE_TRIAL_DATA = False
TRIAL_DATA = { 'model': 'const', 'params': 'Ne', 'data_h5': None,
               'bed_file': None, 'reco_folder': None, 'param_values': '10000.'}

# section C: summary stats customization----------------------------------------
COLOR_DICT = {"YRI": "darkorange","CEU": "blue","CHB": "green", "MXL": "red",
              "simulation": "gray", "msprime": "purple"}

SS_LABELS = []
SS_COLORS = []
'''
Override by commenting out the function body,
and adding in your definitions. Leave the assert
at the end.
'''
def update_ss_labels(pop_names):
    # SS_LABELS is a list of string labels, ex ["CEU", "YRI", "CHB", "simulation"]
    # or ["msprime", "SLiM"]
    if pop_names == '':
        pop_names = 'msprime'
    
    SS_LABELS.extend(pop_names.split("_"))
    SS_LABELS.append("simulation")

    # colors for plotting, ex ["blue", "darkorange", "green", "gray"] (last is traditionally gray)
    for label in SS_LABELS:
        SS_COLORS.append(COLOR_DICT[label])

    assert len(SS_LABELS) == len(SS_COLORS)

# Section D: alternate data format options--------------------------------------
NEW_DATA = False # 1000 Genomes high-coverage data

'''The high-coverage data ("new data") appears to have partial filtering on
singletons. It is recommended, if using the high-coverage data, to enable
singleton filtering for both real and simulated data. It may be necessary to
experiment with different filtering rates.'''
FILTER_SIMULATED = False
FILTER_REAL_DATA = False
FILTER_RATE = 0.50
NUM_SNPS_ADJ = NUM_SNPS * 3
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # testing
    print(SS_LABELS)
    print(SS_COLORS)
    update_ss_labels("CEU")
    print(SS_LABELS)
    print(SS_COLORS)
