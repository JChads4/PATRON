
# Program for the Analysis of Transfermium ROtational Nuclei - PATRON

Simulation tool for analysing coupled rotational band structures in heavy nuclei through gamma-ray and conversion electron spectroscopy.

## Features
- Monte Carlo simulation of nuclear decay
- Internal conversion coefficient calculations (requires BrIcc)
- Gamma-ray and conversion electron spectra generation
- Experimental data comparison and χ² analysis
- Configuration-based workflow

## Installation

1. Clone repository:
```bash
git clone https://github.com/JChads4/PATRON.git
cd PATRON
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install BrIcc (required for conversion coefficients): [BrIcc Download](https://www.nndc.bnl.gov/bricc/)

## Directory Structure
```
ROTBANDSIM/
├── configs/              
│   └── 254No.yaml      # Example configuration
├── data/
│   ├── experimental/    # Your experimental data
│   └── level_schemes/   
│       └── 254No.txt   # Example level scheme
├── src/
│   ├── sim.py          
│   └── utils.py        
└── output/             
```

## Getting Started

1. Example level scheme format (data/level_schemes/254No.txt):
```
# Spin, Level Energy, E2 Energy, M1 Energy, Level Number, Population
8,0,0,0,0,0.001
9,257,0,257,1,0.002
10,531,531,274,2,0.005
...
```

2. Add your experimental data to data/experimental/:
- Gamma spectrum: TEXT file with columns [Energy(keV), Counts]
- Electron spectrum: TEXT file with columns [Energy(keV), Counts]

3. Create configuration (example configs/254No.yaml):
```yaml
nucleus:
  Q0: 12.4        # Deformation parameter
  K: 8            # Bandhead K value
  elem_sym: No    # Element symbol
  z_num: 102      # Atomic number
  mass: 254       # Mass number
  GK_GR_values: [0.] # gK-gR value(s) to test
  total_recoils: 10000000
  gamma_peak: 257  # Peak for normalisation
  gamma_range: 3   # Range for normalisation

files:
  level_scheme: data/level_schemes/254No_isomer.txt
  exp_electron: data/experimental/your_electron_file.dat  # Optional, set 'null' if not required.
  exp_gamma: data/experimental/your_gamma_file.dat        # Optional, set 'null' if not required.

experiment:
  hv_barrier: 25                                              # Electric field barrier for electron spectrum. Nothing will be plotted under this value.
  normalise_spectra: true                                     # Normalisation through comparison of exp/ sim gamma peak areas. 
  show_exp_spectra: true
  elec_eff_params: [1.273, -1.541, -0.943, -0.128, -0.00137]
  gam_eff_params: [1.866, -0.627, -0.201, 0.246, -0.0779]
  elec_fwhm: [0.0040, 5.8762]                                 # [0] = m, [1] = c -> FWHM = m*x + c
  gam_fwhm: [0.0013, 1.8302]                                  # [0] = m, [1] = c -> FWHM = m*x + c

theory:
  gR_vals: [0.4, 0.36, 0.32, 0.28] # Quenched or unquenched values of gR.
  gK_vals: [-0.0225, 1.001]        # Theoretical gK values to plot in the ChiSq plot.
```

4. Run simulation:
```bash
python src/sim.py 254No
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- **BrIcc**: Thanks to the developers of BrIcc for providing the tool for internal conversion coefficient calculations.
- **SciencePlots**: The `scienceplots` package is used for enhanced plotting aesthetics.

## Contact

For questions or feedback, please contact Jamie Chadderton at [jamiechadderton8@gmail.com].
