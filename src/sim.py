
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from sympy import sqrt as sympy_sqrt
import os
from utils import *
import pathlib
from sympy import sqrt as sympy_sqrt
import argparse
import yaml

# Create directories
for dir in ['data/experimental', 'data/level_schemes', 'output']:
    os.makedirs(dir, exist_ok=True)


# def load_config(config_name):
#     with open(f'configs/{config_name}.yaml', 'r') as f:
#         config = yaml.safe_load(f)
        
#     # Evaluate GK_GR expression if it contains calculation
#     if isinstance(config['nucleus']['GK_GR_values'][0], str):
#         z_num = config['nucleus']['z_num']
#         mass = config['nucleus']['mass']
#         expr = config['nucleus']['GK_GR_values'][0].replace('(101/249)', f'({z_num}/{mass})')
#         config['nucleus']['GK_GR_values'] = [eval(expr)]
    
    # return config

def load_config(config_name):
    with open(f'configs/{config_name}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate element symbol is a string
    elem_sym = config['nucleus']['elem_sym']
    if not isinstance(elem_sym, str):
        raise ValueError(f"Element symbol must be a quoted string in config, got {type(elem_sym)}")
    
    if isinstance(config['nucleus']['GK_GR_values'][0], str):
        z_num = config['nucleus']['z_num']
        mass = config['nucleus']['mass']
        expr = config['nucleus']['GK_GR_values'][0].replace('(101/249)', f'({z_num}/{mass})')
        config['nucleus']['GK_GR_values'] = [eval(expr)]
    
    return config

plt.style.use(["science"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Configuration name (e.g. 254No)')
    args = parser.parse_args()
    config = load_config(args.config)

    # Process each GK-GR value
    dfs = {}
    for GK_GR in config['nucleus']['GK_GR_values']:
        label = f"GK_GR_{GK_GR}"
        
        # Read level scheme
        level_scheme_data = np.loadtxt(config['files']['level_scheme'], 
                                    delimiter=',', skiprows=2).T
        I_initial, _, Energy_E2, Energy_M1, Level_energy, norm_pop = level_scheme_data.astype(float)

        # Process transitions
        data = []
        for i, E2, M1, Level_energy, norm_pop in zip(I_initial, Energy_E2, Energy_M1, 
                                                    Level_energy, norm_pop):
            row_data = process_transition(i, E2, M1, Level_energy, norm_pop, GK_GR, config)
            data.append(row_data)

        df = pd.DataFrame(data).applymap(format_significant_figures)
        dfs[label] = df

    # Calculate intensities
    for idx, (label, df) in enumerate(dfs.items()):
        dfs[label] = calculate_intensities(df, config['nucleus']['total_recoils'])

    # Plot spectra
    plot_combined_spectra(
        dfs, 
        config['nucleus']['elem_sym'],
        gamma_bin_width=1,
        electron_bin_width=3,
        gam_eff_params=config['experiment']['gam_eff_params'],
        gam_fwhm=config['experiment']['gam_fwhm'],
        elec_eff_params=config['experiment']['elec_eff_params'],
        elec_fwhm=config['experiment']['elec_fwhm'],
        exp_electron_spectrum=config['files']['exp_electron'],
        exp_gamma_spectrum=config['files']['exp_gamma'],
        gamma_peak=config['nucleus']['gamma_peak'],
        gamma_range=config['nucleus']['gamma_range'],
        show_exp_spectra=config['experiment']['show_exp_spectra'],
        normalise_simulated_spectra=config['experiment']['normalise_spectra'],
        theory_gR_vals=config['theory']['gR_vals'],
        theory_gK_vals=config['theory']['gK_vals']
    )
    plt.show()

if __name__ == '__main__':
    main()
