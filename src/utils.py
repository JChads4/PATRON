import numpy as np
import pandas as pd
from scipy.stats import norm
import xml.etree.ElementTree as ET
import subprocess
from sympy import sqrt as sympy_sqrt
import matplotlib.pyplot as plt
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple
import math

@lru_cache(maxsize=128)
def calculate_cg(i: float, K: float) -> Tuple[float, float, float]:
    """Calculate Clebsch-Gordan coefficients with caching."""
    from sympy import S
    from sympy.physics.quantum.cg import CG
    
    cg_BM1 = CG(S(i), S(K), S(1), S(0), S(i - 1), S(K)).doit()
    cg_BE2_str = CG(S(i), S(K), S(2), S(0), S(i - 2), S(K)).doit()
    cg_BE2_non_str = CG(S(i), S(K), S(2), S(0), S(i - 1), S(K)).doit()
    
    return (float(abs(cg_BM1.evalf())), 
            float(abs(cg_BE2_str.evalf())), 
            float(abs(cg_BE2_non_str.evalf())))

class EnergyLevel:
    __slots__ = ('level', 'population', 'ratio', 'transitions_count')
    
    def __init__(self, level: float, population: float):
        self.level = level
        self.population = float(population)
        self.ratio = {'i-1': 0.0, 'i-2': 0.0}
        self.transitions_count = {'i-1': 0, 'i-2': 0}

    def set_transition(self, transition: str, branching_ratio: float):
        self.ratio[transition] = float(branching_ratio)

def simulate_decay_monteCarlo(levels: List[EnergyLevel]) -> pd.DataFrame:
    """Vectorised Monte Carlo simulation of nuclear decay."""
    decay_data = {'Level': [], 'M1': [], 'E2': []}
    total_nuclei = sum(level.population for level in levels)
    print(f'Total Nuclei at beginning = {int(total_nuclei)}')
    
    for current_level in levels:
        # Vectorized decay decisions
        pop = int(current_level.population)
        if pop > 0:
            decays = np.random.random(pop) < current_level.ratio['i-1']
            m1_decays = np.sum(decays)
            e2_decays = pop - m1_decays
            
            current_level.transitions_count['i-1'] += m1_decays
            current_level.transitions_count['i-2'] += e2_decays
            
            target_m1 = int(current_level.level - 1)
            target_e2 = int(current_level.level - 2)
            
            if 0 <= target_m1 < len(levels):
                levels[target_m1].population += m1_decays
            if 0 <= target_e2 < len(levels):
                levels[target_e2].population += e2_decays
        
        decay_data['Level'].append(current_level.level)
        decay_data['M1'].append(current_level.transitions_count['i-1'])
        decay_data['E2'].append(current_level.transitions_count['i-2'])
        current_level.population = 0

    print(f'Sum of M1s = {sum(decay_data["M1"])}')
    print(f'Sum of E2s = {sum(decay_data["E2"])}')
    return pd.DataFrame(decay_data)[::-1]

def process_transition(i, E2, M1, Level_energy, norm_pop, GK_GR, config):
    """Process single transition calculations."""
    K = config['nucleus']['K']
    Q0 = config['nucleus']['Q0'] 
    elem_sym = config['nucleus']['elem_sym']
    
    cg_BM1, cg_BE2_str, cg_BE2_non_str = calculate_cg(i, K)
    delta = 1e12 if GK_GR == 0 else 0.93 * (M1 / 1000) * Q0 / (abs(GK_GR) * sympy_sqrt((i ** 2) - 1))
    
    alphas = calculate_conversion_coefficients(elem_sym, [M1], [E2], delta)
    alpha_I_min_1 = alphas['I-1_Tot'].get(M1, 0.0)
    alpha_I_min_2 = alphas['I-2_Tot'].get(E2, 0.0)
    
    other_I_min_1 = sum(alphas[f'I-1_{shell}'].get(M1, 0.0) for shell in ['N-tot', 'O-tot', 'P-tot', 'Q-tot'])
    other_I_min_2 = sum(alphas[f'I-2_{shell}'].get(E2, 0.0) for shell in ['N-tot', 'O-tot', 'P-tot', 'Q-tot'])
    
    other_ratio_I_min_1 = other_I_min_1 / alpha_I_min_1 if alpha_I_min_1 != 0 else 0.0
    other_ratio_I_min_2 = other_I_min_2 / alpha_I_min_2 if alpha_I_min_2 != 0 else 0.0

    subshell_alphas = {}
    for shell in ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5']:
        i1_shell = alphas[f'I-1_{shell}'].get(M1, 0.0)
        i2_shell = alphas[f'I-2_{shell}'].get(E2, 0.0)
        subshell_alphas[f'Alpha I-1 {shell}'] = i1_shell
        subshell_alphas[f'Alpha I-2 {shell}'] = i2_shell
        subshell_alphas[f'Alpha i-1 {shell}/Tot'] = i1_shell / alpha_I_min_1 if alpha_I_min_1 != 0 else 0.0
        subshell_alphas[f'Alpha i-2 {shell}/Tot'] = i2_shell / alpha_I_min_2 if alpha_I_min_2 != 0 else 0.0


    BM1 = (3 / (4 * np.pi)) * (cg_BM1 ** 2) * (K ** 2) * ((GK_GR) ** 2)
    BE2_non_str = (5 / (16 * np.pi)) * (cg_BE2_non_str ** 2) * (Q0 ** 2) 
    BE2_str = (5 / (16 * np.pi)) * (cg_BE2_str ** 2) * (Q0 ** 2)
    
    T_M1 = (1.758e13) * ((M1 / 1000) ** 3) * BM1
    T_E2_non_str = (1.225e9) * ((M1 / 1000) ** 5) * BE2_non_str * 10000
    T_E2_str = (1.225e9) * ((E2 / 1000) ** 5) * BE2_str * 10000
    
    T_i_min_1 = (T_M1 + T_E2_non_str) * (1 + alpha_I_min_1)  
    T_i_min_2 = T_E2_str * (1 + alpha_I_min_2)
    
    BR_i_min_1 = T_i_min_1 / (T_i_min_1 + T_i_min_2)
    BR_i_min_2 = T_i_min_2 / (T_i_min_1 + T_i_min_2)

    return_dict = {
        'Initial Spin': i,
        'Level Energy': Level_energy,
        'norm_pop': norm_pop,
        'Energy E2 (keV)': E2,
        'Energy M1 (keV)': M1,
        'BM1': BM1,
        'BE2_non_str': BE2_non_str,
        'BE2_str': BE2_str,
        'T_M1': T_M1,
        'T_E2_non_str': T_E2_non_str,
        'T_E2_str': T_E2_str,
        'BR_I_min_1': BR_i_min_1,
        'BR_I_min_2': BR_i_min_2,
        'delta': delta,
        'Alpha I-1': alpha_I_min_1,
        'Alpha I-2': alpha_I_min_2,
        'Label': f"GK_GR_{GK_GR}"
    }

    return_dict.update(subshell_alphas)
    return return_dict

def norm_gaussian(x: float, a: float, mu: float, sigma: float) -> float:
    """Calculate normalized Gaussian."""
    return a * (1/(sigma * np.sqrt(2 * math.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def calculate_entry_distribution(levels: List[float], params: Tuple[float, float, float]) -> Dict[float, float]:
    """Calculate the entry distribution using specified parameters."""
    a, mu, sigma = params
    return {level: norm_gaussian(level, a, mu, sigma)/2.14 for level in levels}

def add_recoils_to_df(df: pd.DataFrame, total_recoils: int) -> pd.DataFrame:
    """Add recoil calculations to DataFrame."""
    df['recoils'] = df['norm_pop'] * total_recoils
    df['population'] = df['recoils']
    return df

@lru_cache(maxsize=256)
def calculate_conversion_coefficients(elem: str, M1_energies: List[float], 
                                   E2_energies: List[float], delta: float) -> Dict:
    """Calculate internal conversion coefficients using BrIcc with caching."""
    shells = ['Tot', 'K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 
              'N-tot', 'O-tot', 'P-tot', 'Q-tot']
    alphas = {f'I-1_{shell}': {} for shell in shells}
    alphas.update({f'I-2_{shell}': {} for shell in shells})

    try:
        for M1_energy in M1_energies:
            if M1_energy > 0:
                cmd = f'briccs -S {elem} -g {M1_energy} -d {delta:.4f} -L M1+E2 -a'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    shell_alphas = parse_briccs_output(result.stdout)
                    for shell in shells:
                        alphas[f'I-1_{shell}'][M1_energy] = shell_alphas.get(shell, 0.0)
        
        for E2_energy in E2_energies:
            if E2_energy > 0:
                cmd = f'briccs -S {elem} -g {E2_energy} -L E2 -a'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    shell_alphas = parse_briccs_output(result.stdout)
                    for shell in shells:
                        alphas[f'I-2_{shell}'][E2_energy] = shell_alphas.get(shell, 0.0)
    except subprocess.SubprocessError as e:
        print(f"BrIcc calculation failed: {e}")
    
    return alphas

def calculate_conversion_coefficients_subshells(element: str, energy: float, 
                                             multipolarity: str, delta: float) -> Dict:
    """Calculate conversion coefficients for individual subshells."""
    shells = ['Tot', 'K', 'L-tot', 'L1', 'L2', 'L3', 'M-tot', 'M1', 'M2', 'M3', 'M4', 'M5', 
             'N-tot', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O-tot', 'O1', 'O2', 'O3', 
             'O4', 'O5', 'P-tot', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']
    
    alphas = {shell: {'alpha': 0.0, 'Eic': 0.0} for shell in shells}
    
    try:
        if multipolarity == 'E2':
            cmd = f'briccs -S {element} -g {energy} -L E2 -a'
        elif multipolarity == 'M1':
            cmd = f'briccs -S {element} -g {energy} -L M1 -a'
        elif multipolarity == 'E1':
            cmd = f'briccs -S {element} -g {energy} -L E1 -a'
        elif multipolarity == 'M1+E2':
            cmd = f'briccs -S {element} -g {energy} -d {delta:.4f} -L M1+E2 -a'
        else:
            raise ValueError('Invalid multipolarity')

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            alphas.update(parse_briccs_output_subshells(result.stdout, multipolarity))
    
    except subprocess.SubprocessError as e:
        print(f"BrIcc calculation failed: {e}")
    
    return alphas

def parse_briccs_output_subshells(output: str, multipolarity: str) -> Dict:
    """Parse BrIcc XML output for subshell coefficients."""
    root = ET.fromstring(output)
    shells = ['Tot', 'K', 'L-tot', 'L1', 'L2', 'L3', 'M-tot', 'M1', 'M2', 'M3', 'M4', 'M5', 
             'N-tot', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O-tot', 'O1', 'O2', 'O3', 
             'O4', 'O5', 'P-tot', 'P1', 'P2', 'P3', 'P4', 'P5', 'Q']
    
    alphas = {shell: {'alpha': 0.0, 'Eic': 0.0} for shell in shells}
    
    xpath = './/MixedCC' if multipolarity == 'M1+E2' else './/PureCC'
    for cc in root.findall(xpath):
        shell = cc.get('Shell')
        if shell in shells:
            alphas[shell]['alpha'] = float(cc.text.strip())
            alphas[shell]['Eic'] = float(cc.get('Eic', 0.0))
    
    return alphas

def parse_briccs_output(output: str) -> Dict:
    """Parse BrIcc XML output for total coefficients."""
    root = ET.fromstring(output)
    shells = ['Tot', 'K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 
             'N-tot', 'O-tot', 'P-tot', 'Q-tot']
    
    alphas = {shell: 0.0 for shell in shells}
    
    for purecc in root.findall('.//PureCC'):
        shell = purecc.get('Shell')
        if shell in shells:
            alphas[shell] = float(purecc.text.strip())
    
    for mixedcc in root.findall('.//MixedCC'):
        shell = mixedcc.get('Shell')
        if shell in shells:
            alphas[shell] = float(mixedcc.text.strip())
    
    return alphas

def format_significant_figures(val) -> str:
    """Format to 4 significant figures."""
    try:
        return f"{float(val):.4g}"
    except (ValueError, TypeError):
        return val

def fwhm(energy: float, *params) -> float:
    """Calculate FWHM."""
    return params[0]*energy + params[1]

def fwhm_to_sigma(fwhm: float) -> float:
    """Convert FWHM to sigma."""
    return fwhm/2.355

def elec_efficiency(energy: float, a: float, b: float, c: float, d: float, e: float) -> float:
    """Calculate electron efficiency."""
    x = np.log(energy/320)
    eff = np.exp(a + (b*x) + (c*x**2) + (d*x**3) + (e*x**4))
    return eff/100

def gamma_efficiency(energy: float, a: float, b: float, c: float, d: float, e: float) -> float:
    """Calculate gamma efficiency."""
    x = np.log(energy/320)
    eff = np.exp(a + (b*x) + (c*x**2) + (d*x**3) + (e*x**4))
    return eff/100

def err_elec_efficiency(eff: float) -> float:
    """Calculate electron efficiency error."""
    return 0.1 * eff

def err_gamma_efficiency(eff: float) -> float:
    """Calculate gamma efficiency error."""
    return 0.1 * eff

def gaussian(x: np.ndarray, area: float, mean: float, sigma: float) -> np.ndarray:
    """Calculate Gaussian distribution."""
    return area * norm.pdf(x, loc=mean, scale=sigma)

def sum_counts_in_range(counts: np.ndarray, bins: np.ndarray, 
                       energy_range: Tuple[float, float]) -> float:
    """Sum counts within energy range."""
    range_mask = (bins >= energy_range[0]) & (bins < energy_range[1])
    return np.sum(counts[range_mask])

def calculate_chisq(observed: np.ndarray, expected: np.ndarray) -> float:
    """Calculate reduced chi-squared value."""
    mask = expected > 0
    observed, expected = observed[mask], expected[mask]
    chisq = np.sum(((observed - expected) ** 2) / expected)
    return chisq / len(observed)

def plot_combined_spectra(dfs, elem_sym, gamma_bin_width=2, electron_bin_width=3, 
                         gam_eff_params = [1.866, -0.627, -0.201, 0.246, -0.0779] , gam_fwhm = [0.0013, 1.8302], 
                         elec_eff_params = [1.273,-1.541, -0.943, -0.128, -0.00137], elec_fwhm= [0.0040, 5.8762],
                         exp_electron_spectrum = None, exp_gamma_spectrum = None,
                         hv_barrier =25,
                         gamma_peak = 120, gamma_range = 5,
                         show_exp_spectra = False, normalise_simulated_spectra=False,
                         theory_gR_vals=[0.4, 0.36, 0.32, 0.28], theory_gK_vals=[-0.0225, 1.001]):
   
   theory_combinations = [(gK - gR, gK, gR) for gK in theory_gK_vals for gR in theory_gR_vals]
   cols =['r', 'b', 'g', 'y']
   gK_colors = {gK: color for gK, color in zip(theory_gK_vals, cols)}
   gR_line_styles = {gR: linestyle for gR, linestyle in zip(theory_gR_vals, ['-', '--', '-.', ':'])}

   width = 12
   fig, (ax_gamma, ax_electron) = plt.subplots(2, 1, figsize=(width, width*3))
   fig.patch.set_alpha(0.)
   plt.subplots_adjust(hspace=0.5)
   fs = 22

   # Color for the plot
   colour = ['r', 'b', 'g', 'purple', 'darkorange', 'teal', 'darkred', 'lightsteelblue', 'hotpink', 'lawngreen', 'chocolate', 'dimgrey',
           'bisque', 'cornflowerblue', 'lavender', 'mediumorchid', 'rosybrown', 'darkkhaki', 'cadetblue', 'darkolivegreen', 'pink', 
           'coral', 'lime', 'tan', 'fuchsia']

   total_energy_range = np.linspace(0, 1000, 10000)

   # CHISQ PLOTTING LISTS
   gk_gr_chisq_values = []
   elec_chisq_values = []
   gam_chisq_values = []

   # Loop through each DataFrame in the dictionary
   for idx, (label, df) in enumerate(dfs.items()):
       color = colour[idx]
       
       total_gamma_intensity = np.zeros_like(total_energy_range)
       total_electron_intensity = np.zeros_like(total_energy_range)
       for i, row in df.iterrows():
           # Gamma Spectra
           g_intensity_i_1 = pd.to_numeric(row['Gamma Intensity I-1'], errors='coerce')
           g_intensity_i_2 = pd.to_numeric(row['Gamma Intensity I-2'], errors='coerce')
           g_energy_E2 = pd.to_numeric(row['Energy E2 (keV)'])
           g_energy_M1 = pd.to_numeric(row['Energy M1 (keV)'])

           g_E2_eff = gamma_efficiency(g_energy_E2, *gam_eff_params) if g_energy_E2 > 0 else 0
           g_M1_eff = gamma_efficiency(g_energy_M1, *gam_eff_params)
           g_intensity_i_2 = g_E2_eff * g_intensity_i_2
           g_intensity_i_1 = g_M1_eff * g_intensity_i_1

           g_fwhm_m1 = fwhm(g_energy_M1, *gam_fwhm)
           g_sigma_m1 = fwhm_to_sigma(g_fwhm_m1)
           g_M1_area = gamma_bin_width * g_intensity_i_1
           g_M1_range = np.linspace(g_energy_M1 - 30, g_energy_M1 + 30, 1000)
           g_M1_peak = gaussian(g_M1_range, g_M1_area, g_energy_M1, g_sigma_m1)
           total_gamma_intensity += np.interp(total_energy_range, g_M1_range, g_M1_peak, left=0, right=0)

           g_fwhm_e2 = fwhm(g_energy_E2, *gam_fwhm)
           g_sigma_e2 = fwhm_to_sigma(g_fwhm_e2)
           g_E2_area = gamma_bin_width * g_intensity_i_2
           g_E2_range = np.linspace(g_energy_E2 - 30, g_energy_E2 + 30, 1000)
           g_E2_peak = gaussian(g_E2_range, g_E2_area, g_energy_E2, g_sigma_e2)
           total_gamma_intensity += np.interp(total_energy_range, g_E2_range, g_E2_peak, left=0, right=0)

           # Electron Spectra
           e_intensity_i_1 = pd.to_numeric(row['Electron Intensity I-1'], errors='coerce')
           e_intensity_i_2 = pd.to_numeric(row['Electron Intensity I-2'], errors='coerce')

           subshells = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5']
           
           for shell in subshells:
               alphas_m1 = calculate_conversion_coefficients_subshells(elem_sym, g_energy_M1, 'M1', row['Label'].rstrip('GK_GR_'))
               electron_energy_m1 = alphas_m1[shell]['Eic']
               subshell_ratio_i_1 = pd.to_numeric(row[f'Alpha i-1 {shell}/Tot'])
               e_subshell_intensity_i_1 = e_intensity_i_1 * subshell_ratio_i_1
               e_subshell_intensity_i_1 = e_subshell_intensity_i_1 * elec_efficiency(electron_energy_m1, *elec_eff_params)  if electron_energy_m1 > 0 else 0
               elec_fwhm_m1 = fwhm(electron_energy_m1, *elec_fwhm)
               elec_sigma_m1 = fwhm_to_sigma(elec_fwhm_m1)
               e_M1_area = electron_bin_width * e_subshell_intensity_i_1
               e_M1_range = np.linspace(electron_energy_m1 - 30, electron_energy_m1 + 30, 1000)
               e_M1_peak = gaussian(e_M1_range, e_M1_area, electron_energy_m1, elec_sigma_m1)
               total_electron_intensity += np.interp(total_energy_range, e_M1_range, e_M1_peak, left=0, right=0)

               alphas_e2 = calculate_conversion_coefficients_subshells(elem_sym, g_energy_E2, 'E2', row['Label'].rstrip('GK_GR_'))
               electron_energy_e2 = alphas_e2[shell]['Eic']
               subshell_ratio_i_2 = pd.to_numeric(row[f'Alpha i-2 {shell}/Tot'])
               e_subshell_intensity_i_2 = e_intensity_i_2 * subshell_ratio_i_2
               e_subshell_intensity_i_2 = e_subshell_intensity_i_2 * elec_efficiency(electron_energy_e2, *elec_eff_params)  if electron_energy_e2 > 0 else 0
               elec_fwhm_e2 = fwhm(electron_energy_e2, *elec_fwhm)
               elec_sigma_e2 = fwhm_to_sigma(elec_fwhm_e2)
               e_E2_area = electron_bin_width * e_subshell_intensity_i_2
               e_E2_range = np.linspace(electron_energy_e2 - 30, electron_energy_e2 + 30, 1000)
               e_E2_peak = gaussian(e_E2_range, e_E2_area, electron_energy_e2, elec_sigma_e2)
               total_electron_intensity += np.interp(total_energy_range, e_E2_range, e_E2_peak, left=0, right=0)

       # Gamma histogram
       gamma_bin_edges = np.arange(0, 500 + gamma_bin_width, gamma_bin_width)
       gamma_binned_intensity, _ = np.histogram(total_energy_range, bins=gamma_bin_edges, weights=total_gamma_intensity)
       gamma_bin_centers = 0.5 * (gamma_bin_edges[:-1] + gamma_bin_edges[1:])

       # Electron histogram
       electron_bin_edges = np.arange(0, 500 + electron_bin_width, electron_bin_width)
       electron_binned_intensity, _ = np.histogram(total_energy_range, bins=electron_bin_edges, weights=total_electron_intensity)
       electron_bin_centers = 0.5 * (electron_bin_edges[:-1] + electron_bin_edges[1:])
           
       # LOAD ELECTRONS
       if show_exp_spectra:
           energy, counts = None, None
           genergy, gcounts = None, None
           
           if exp_electron_spectrum:
               data = np.loadtxt(exp_electron_spectrum)
               energy, counts = data[:, 0], data[:, 1]
               counts[counts < 0] = 0
               mask = energy > hv_barrier
               energy, counts = energy[mask], counts[mask]
               
               # CHISQ STUFF for electrons    
               exp_binned, _ = np.histogram(energy, bins=electron_bin_edges, weights=counts)
               elec_chisq = calculate_chisq(exp_binned, electron_binned_intensity)
               elec_chisq_values.append(elec_chisq)
           
           if exp_gamma_spectrum:
               # LOAD GAMMAS
               gdata = np.loadtxt(exp_gamma_spectrum)
               genergy, gcounts = gdata[:, 0], gdata[:, 1]
               gcounts[gcounts < 0] = 0
               
               # CHISQ for gammas
               gam_exp_binned, _ = np.histogram(genergy, bins=gamma_bin_edges, weights=gcounts)
               gam_chisq = calculate_chisq(gam_exp_binned, gamma_binned_intensity)
               gam_chisq_values.append(gam_chisq)
           
           gk_gr_value = float(label.split('_')[-1])
           gk_gr_chisq_values.append(gk_gr_value)
           
           # PERFORM NORMALISATION
           if normalise_simulated_spectra and exp_gamma_spectrum:
               norm_gamma_counts = sum_counts_in_range(gcounts, genergy, (gamma_peak-gamma_range, gamma_peak+gamma_range))
               print(norm_gamma_counts)
               norm_gamma_area = norm_gamma_counts
               print(norm_gamma_counts, norm_gamma_area)
               g_bins, gamma_binned_intensity, e_bins, electron_binned_intensity = normalise_spectra_by_gamma_area(
                   electron_bin_edges, electron_binned_intensity, gamma_bin_edges, gamma_binned_intensity, 
                   gamma_peak, gamma_range, norm_gamma_area, f'{gk_gr_value:.2f}'
               )

       ax_electron.step(electron_bin_centers[electron_bin_centers>hv_barrier], 
                       electron_binned_intensity[electron_bin_centers>hv_barrier], 
                       where='pre', color=color, linewidth=1, 
                       label=r'($gK$-$gR$) = {:.2f}'.format(gk_gr_value))
       ax_electron.fill_between(electron_bin_centers[electron_bin_centers>hv_barrier], 
                              electron_binned_intensity[electron_bin_centers>hv_barrier], 
                              step="pre", color=color, alpha=0.2)
                              
       ax_gamma.step(gamma_bin_centers[gamma_bin_centers>hv_barrier], 
                    gamma_binned_intensity[gamma_bin_centers>hv_barrier], 
                    where='pre', color=color, linewidth=1, 
                    label=r'$(gK-gR) = {:.2f}$'.format(gk_gr_value))
       ax_gamma.fill_between(gamma_bin_centers, gamma_binned_intensity, 
                           step="pre", color=color, alpha=0.2)

   # Configure axes
   ax_gamma.set_xlabel('Energy (keV)', fontsize=fs)
   ax_gamma.set_ylabel(f'Intensity / {gamma_bin_width} keV', fontsize=fs)
   ax_gamma.set_xlim(0,1000)
   ax_gamma.set_title('Gamma Spectra', fontsize=fs)
   ax_gamma.legend(loc='upper right', fontsize=fs - 2)
   ax_gamma.tick_params(axis='both', which='major', labelsize=fs - 2)

   ax_electron.set_xlabel('Energy (keV)', fontsize=fs)
   ax_electron.set_ylabel(f'Intensity / {electron_bin_width} keV', fontsize=fs)
   ax_electron.set_ylim(bottom=0)
   ax_electron.set_title('Simulated Electron Spectra', fontsize=fs)
   ax_electron.legend(loc='upper right', fontsize=fs - 2)
   ax_electron.tick_params(axis='both', which='major', labelsize=fs - 2)

   # Plot experimental data if available
   if show_exp_spectra:
       if exp_electron_spectrum and energy is not None and counts is not None:
           ax_electron.stairs(counts, np.append(energy, energy[-1] + np.diff(energy)[-1]), 
                            lw=1, color='k', label='Experiment')
       
       if exp_gamma_spectrum and genergy is not None and gcounts is not None:
           ax_gamma.stairs(gcounts, np.append(genergy, genergy[-1] + np.diff(genergy)[-1]), 
                         fill=False, lw=1, color='k', label='Experiment')

       # Plot chi-squared values
       if gam_chisq_values or elec_chisq_values:
           width = 8
           chisq_fig, (ax_chisq_gamma, ax_chisq_electron) = plt.subplots(2, 1, figsize=(width, width / 1.2))
           plt.subplots_adjust(hspace=0.4)
           
           if gam_chisq_values:
                ax_chisq_gamma.tick_params(axis='both', which='major', labelsize=fs - 4)
                ax_chisq_gamma.set_xlabel(r'$gK - gR$', fontsize=fs - 2)
                ax_chisq_gamma.set_ylabel(r'$\chi_{r}^2$', fontsize=fs - 2)
                ax_chisq_gamma.set_ylim(bottom=0)
                ax_chisq_gamma.set_title('Gammas', fontsize=fs)
                ax_chisq_gamma.legend(loc='upper right', fontsize=fs - 2)
                ax_chisq_gamma.plot(gk_gr_chisq_values, gam_chisq_values, marker='o', 
                                    linestyle='--', linewidth=2, color='k')
                ax_chisq_gamma.axhline(y=1, color='g', linestyle='-.', linewidth=3)

           if elec_chisq_values:
                ax_chisq_electron.tick_params(axis='both', which='major', labelsize=fs - 4)
                ax_chisq_electron.set_xlabel(r'$gK - gR$', fontsize=fs - 2)
                ax_chisq_electron.set_ylabel(r'$\chi_{r}^2$', fontsize=fs - 2)
                ax_chisq_electron.set_ylim(bottom=0)
                ax_chisq_electron.set_title('Electrons', fontsize=fs)
                ax_chisq_electron.legend(loc='upper right', fontsize=fs - 2)
                ax_chisq_electron.plot(gk_gr_chisq_values, elec_chisq_values, marker='o', 
                                    linestyle='--', linewidth=2, color='k')
                ax_chisq_electron.axhline(y=1, color='g', linestyle='-.', linewidth=3)

            # Plot theory lines
           for (diff, gK, gR) in theory_combinations:
                color = gK_colors[gK]
                linestyle = gR_line_styles[gR]
                if gam_chisq_values:
                    ax_chisq_gamma.axvline(diff, color=color, linestyle=linestyle)
                if elec_chisq_values:
                    ax_chisq_electron.axvline(diff, color=color, linestyle=linestyle)

           plt.legend(fontsize=fs - 2)



def normalise_electron_spectra(e_bins, e_counts, norm_value):

    simulated_e_bins = e_bins
    simulated_e_counts = e_counts

    # Integrate the counts of the simulated electron spectrum to get the total area
    total_area_simulated = np.sum(simulated_e_counts * np.diff(simulated_e_bins))
    
    # Calculate the normalisation factor
    norm_factor = norm_value / total_area_simulated
    print(f'############ ELECTRON NORMALISATION FACTOR = {norm_factor} #################')
    
    # Normalise the counts of the simulated spectrum
    normalised_e_counts = simulated_e_counts * norm_factor

    print(f"Sum of sim counts = {np.sum(normalised_e_counts)}, Total electron exp counts = {norm_value}")
    
    return normalised_e_counts

def normalise_spectra(e_bins, e_counts, g_counts, norm_value):

    simulated_e_bins = e_bins
    simulated_e_counts = e_counts
    simulated_g_counts = g_counts

    # Integrate the counts of the simulated electron spectrum to get the total area
    total_area_simulated = np.sum(simulated_e_counts * np.diff(simulated_e_bins))
    
    # Calculate the normalisation factor
    norm_factor = norm_value / total_area_simulated
    print(f'############ ELECTRON NORMALISATION FACTOR = {norm_factor} #################')
    
    # Normalise the counts of the simulated spectrum
    normalised_e_counts = simulated_e_counts * norm_factor
    normalised_g_counts = simulated_g_counts * norm_factor

    print(f"Sum of sim counts = {np.sum(normalised_e_counts)}, Total electron exp counts = {norm_value}")
    
    return normalised_e_counts, normalised_g_counts

def normalise_spectra_by_recoils(sim_e_counts, sim_g_counts, sim_recoils, exp_recoils):

    # Calculate the normalisation factor
    norm_factor = exp_recoils / sim_recoils

    # Normalise the counts of the simulated spectrum
    normalised_e_counts = sim_e_counts * norm_factor
    normalised_g_counts = sim_g_counts * norm_factor
    
    return normalised_e_counts, normalised_g_counts

def normalise_spectra_by_gamma_area(e_bins, e_counts, g_bins, g_counts, peak, gamma_range, norm_value, label):

    sim_counts = sum_counts_in_range(g_counts, g_bins[:-1], (peak-gamma_range, peak+gamma_range))
    background_sim_counts_left = sum_counts_in_range(g_counts, g_bins[:-1], (peak-gamma_range-(gamma_range), peak-gamma_range))
    background_sim_counts_right = sum_counts_in_range(g_counts, g_bins[:-1], (peak+gamma_range, peak+gamma_range+(gamma_range)))
    background_sim_counts = background_sim_counts_left + background_sim_counts_right
    sim_area = (sim_counts - background_sim_counts) * np.diff(g_bins)[0]
    print(f"g_bins = {np.diff(g_bins)[0]}, sim_area = {sim_area}, exp area = {norm_value:.2f}")

    fig, axs = plt.subplots(1, 1, figsize=(16, 12))
    axs.stairs(g_counts, g_bins, label='Normalised Gamma Counts')
    axs.axvspan(peak-gamma_range-(gamma_range), peak-gamma_range, color='b', alpha=0.2, label='Background region')
    axs.axvspan(peak+gamma_range, peak+gamma_range+(gamma_range), color='b', alpha=0.2)
    axs.axvspan(peak-gamma_range, peak+gamma_range, color='r', alpha=0.1)
    axs.set_xlim([peak-20, peak+20])
    output_file = f"output/{label}_normalisation.png"
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig) 
    
    # norm_factor = norm_value / sim_peak_counts
    norm_factor = norm_value / sim_area
    print(f'############ {label} NORMALISATION FACTOR = {norm_factor} #################')
    g_counts = g_counts * norm_factor
    e_counts = e_counts * norm_factor

    return g_bins, g_counts, e_bins, e_counts



def calculate_conversion_coefficients(elem, M1_energies, E2_energies, delta):
    """
    Calculate the conversion coefficients using BrIcc for the specified energies.
    
    Parameters:
    delta: degree of mixing for I-1 transitions.
    M1_energies (list of float): Energies of the M1 transitions.
    E2_energies (list of float): Energies of the E2 transitions.
    
    Returns:
    dict: Dictionary with total conversion coefficients for each energy and transition type.
    """
    shells = ['Tot', 'K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N-tot', 'O-tot', 'P-tot', 'Q-tot']
    alphas = {f'I-1_{shell}': {} for shell in shells}
    alphas.update({f'I-2_{shell}': {} for shell in shells})

    for M1_energy in M1_energies:
        # I-1 case: must be a mixed transition (M1+E2), and need to specify the degree of mixing (delta)
        cmd = f'briccs -S {elem} -g {M1_energy} -d {delta:.4f} -L M1+E2 -a'
        # print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        shell_alphas = parse_briccs_output(result.stdout)
        for shell in shells:
            alphas[f'I-1_{shell}'][M1_energy] = shell_alphas.get(shell, 0.0)
        
    for E2_energy in E2_energies:
        # E2 stretched case
        cmd = f'briccs -S {elem} -g {E2_energy} -L E2 -a'
        # print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        shell_alphas = parse_briccs_output(result.stdout)
        for shell in shells:
            alphas[f'I-2_{shell}'][E2_energy] = shell_alphas.get(shell, 0.0)
    
    return alphas

def parse_briccs_output(output):
    """
    Parse the BrIcc XML output to extract the total conversion coefficients for all relevant shells.
    
    Parameters:
    output (str): XML output from BrIcc.
    
    Returns:
    dict: Dictionary with conversion coefficients for each shell.
    """
    root = ET.fromstring(output)
    shells = ['Tot', 'K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N-tot', 'O-tot', 'P-tot', 'Q-tot']
    alphas = {shell: 0.0 for shell in shells}

    # For the stretched E2 (I-2) transitions
    for purecc in root.findall('.//PureCC'):
        shell = purecc.get('Shell')
        if shell in shells:
            alphas[shell] = float(purecc.text.strip())
    
    # For the mixed M1+E2 (I-1) transitions
    for mixedcc in root.findall('.//MixedCC'):
        shell = mixedcc.get('Shell')
        if shell in shells:
            alphas[shell] = float(mixedcc.text.strip())
    
    return alphas

def format_significant_figures(val):
    """
    Format a value to 4 significant figures with scientific notation where appropriate.
    
    Parameters:
    val: Value to be formatted.
    
    Returns:
    str: Formatted string with 4 significant figures.
    """
    try:
        return f"{float(val):.4g}"
    except ValueError:
        return val
    


def calculate_intensities(df, total_recoils):
    def read_levels_from_df(branching_data):
        levels = []
        for _, row in branching_data.iterrows():
            level = EnergyLevel(row['Initial Spin'], row['recoils'])
            level.set_transition('i-1', row['BR_I_min_1'])
            level.set_transition('i-2', row['BR_I_min_2'])
            levels.append(level)
        levels = levels[::-1]
        return levels

    # Ensure necessary columns are numeric
    df['Initial Spin'] = pd.to_numeric(df['Initial Spin'], errors='coerce')
    df['norm_pop'] = pd.to_numeric(df['norm_pop'], errors='coerce')
    df['BR_I_min_1'] = pd.to_numeric(df['BR_I_min_1'], errors='coerce')
    df['BR_I_min_2'] = pd.to_numeric(df['BR_I_min_2'], errors='coerce')

    # Calculate entry distribution and add recoils
    levels = df['Initial Spin'].unique()
    df = add_recoils_to_df(df, total_recoils)

    # Read levels and simulate decay
    levels = read_levels_from_df(df)
    decay_df = simulate_decay_monteCarlo(levels)

    # Ammend the conv coeffs
    df['Alpha I-1'] = pd.to_numeric(df['Alpha I-1'], errors='coerce')
    df['Alpha I-2'] = pd.to_numeric(df['Alpha I-2'], errors='coerce')

    decay_df = decay_df.rename(columns={'Level': 'Initial Spin'})
    df = df.merge(decay_df[['Initial Spin', 'M1', 'E2']], on='Initial Spin', how='left')

    # Now we calculate the corrected values
    df['Gamma Intensity I-1'] = df['M1'] / (1+df['Alpha I-1'])
    df['Gamma Intensity I-2'] = df['E2'] / (1+df['Alpha I-2'])

    df['Electron Intensity I-1'] = df['Gamma Intensity I-1'] * df['Alpha I-1']
    df['Electron Intensity I-2'] = df['Gamma Intensity I-2'] * df['Alpha I-2']

    # Now we should normalise the intensities, or not.
    # norm_value = 1

    return df

