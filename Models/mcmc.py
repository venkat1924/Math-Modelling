#CODE
# Phase 1: Setup and Forward Model Definition
# import pytensor
# pytensor.config.device = 'cuda'  # Use GPU
#pytensor.config.floatX = 'float32'  # Optional: faster but less precise
#print(pytensor.config.device)  # Should output "cuda"
# 1. Environment Setup
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import ODEintWarning
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pytensor.tensor as pt # ****** ADD THIS IMPORT ******
import warnings
import os
import pymc as pm


print(f"Running on PyMC v{pm.__version__}")
print(f"Running on ArviZ v{az.__version__}")

OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"Plots and inference data will be saved to '{OUTPUT_DIR}/' directory.")

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ODEintWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 2. Define the Forward ODE Model
def prey_predator_ode_system_internal(t, S, params_tuple):
    r = params_tuple[0]
    p1 = params_tuple[1]
    p2 = params_tuple[2]
    d1 = params_tuple[3]
    b = params_tuple[4]
    c1 = params_tuple[5]
    a = params_tuple[6]
    m = params_tuple[7]
    h1 = params_tuple[8]
    d2 = params_tuple[9]
    d3 = params_tuple[10]
    lambda1 = params_tuple[11]
    q = params_tuple[12]
    
    X, Y, Z = S[0], S[1], S[2]
    
    # ****** MODIFICATION: Use pt.maximum instead of np.maximum ******
    denominator1 = pt.maximum(1 + p1 * Y + p2 * Z, 1e-12)
    denominator2 = pt.maximum(a + (1 - m) * X, 1e-12)
    
    dXdt = (r * X) / denominator1 - d1 * X - b * X**2 - (c1 * (1 - m) * X * (Y + q * Z)) / denominator2
    dYdt = (h1 * (1 - m) * X * (Y + q * Z)) / denominator2 - d2 * Y - lambda1 * Y * Z
    dZdt = lambda1 * Y * Z - (d2 + d3) * Z
    return [dXdt, dYdt, dZdt]

# 3. Create an ODE Solver Wrapper for PyMC
def ode_func_for_pymc_wrapper(y, t, p_tuple):
    return prey_predator_ode_system_internal(t, y, p_tuple)


# Phase 2: Data and Probabilistic Model

# 4. Load and Prepare Observed Data
try:
    full_data = pd.read_csv('/home/csegpuserver/mmEL/simulation_results.csv')
except FileNotFoundError:
    print("Error: `simulation_results.csv` not found.")
    print("Please generate this file using the simulation script provided in the problem description.")
    print("Exiting.")
    exit()

if 'run_id' in full_data.columns:
    if full_data['run_id'].dtype == float:
        observed_df = full_data[full_data['run_id'] == 1.0].copy()
    else:
        observed_df = full_data[full_data['run_id'] == 1].copy()
else:
    print("Error: 'run_id' column not found in simulation_results.csv. Exiting.")
    exit()

if observed_df.empty:
    print("Error: No data found for run_id = 1 in simulation_results.csv.")
    print("Ensure the simulation script was run and produced data for this run_id. Exiting.")
    exit()

t_observed = observed_df['t'].values
X_observed = observed_df['X_noisy'].values
Y_observed = observed_df['Y_noisy'].values
Z_observed = observed_df['Z_noisy'].values
S0_observed = np.array([X_observed[0], Y_observed[0], Z_observed[0]])

param_cols = ['r', 'p1', 'p2', 'd1', 'b', 'c1', 'a', 'm', 'h1', 'd2', 'd3', 'lambda1', 'q']
true_params_series = None
if all(col in observed_df.columns for col in param_cols):
    true_params_series = observed_df.iloc[0][param_cols]
    print("True parameters for the loaded trajectory (run_id=1):")
    print(true_params_series)
else:
    print("True parameter columns not found in CSV. Cannot display them for plot reference lines.")


# 5. Specify the Probabilistic Model in PyMC
with pm.Model() as ode_model:
    r = pm.LogNormal('r', mu=np.log(0.5), sigma=0.5) 
    p1 = pm.HalfNormal('p1', sigma=0.5)
    p2 = pm.HalfNormal('p2', sigma=0.5)
    d1 = pm.LogNormal('d1', mu=np.log(0.1), sigma=0.5)
    b = pm.LogNormal('b', mu=np.log(0.05), sigma=0.5)
    c1 = pm.LogNormal('c1', mu=np.log(0.2), sigma=0.5)
    a = pm.LogNormal('a', mu=np.log(0.8), sigma=0.5)
    m = pm.Beta('m', alpha=2.0, beta=2.0)
    h1 = pm.LogNormal('h1', mu=np.log(0.25), sigma=0.5)
    d2 = pm.LogNormal('d2', mu=np.log(0.08), sigma=0.5)
    d3 = pm.LogNormal('d3', mu=np.log(0.03), sigma=0.5)
    lambda1 = pm.LogNormal('lambda1', mu=np.log(0.7), sigma=0.5)
    q = pm.HalfNormal('q', sigma=0.8)

    ode_params = (r, p1, p2, d1, b, c1, a, m, h1, d2, d3, lambda1, q)

    sigma_X_std_val = np.std(X_observed) if np.std(X_observed) > 1e-6 else 0.1
    sigma_Y_std_val = np.std(Y_observed) if np.std(Y_observed) > 1e-6 else 0.1
    sigma_Z_std_val = np.std(Z_observed) if np.std(Z_observed) > 1e-6 else 0.1

    sigma_X = pm.HalfNormal('sigma_X', sigma=0.25 * sigma_X_std_val)
    sigma_Y = pm.HalfNormal('sigma_Y', sigma=0.25 * sigma_Y_std_val)
    sigma_Z = pm.HalfNormal('sigma_Z', sigma=0.25 * sigma_Z_std_val)

    ode_solution_model = pm.ode.DifferentialEquation(
        func=ode_func_for_pymc_wrapper,
        times=t_observed,
        n_states=3,
        n_theta=len(ode_params),
        t0=0.0,
    )

    ode_solutions = ode_solution_model(
        y0=S0_observed,
        theta=ode_params
    )

    pm.Normal('X_obs', mu=ode_solutions[:, 0], sigma=sigma_X, observed=X_observed)
    pm.Normal('Y_obs', mu=ode_solutions[:, 1], sigma=sigma_Y, observed=Y_observed)
    pm.Normal('Z_obs', mu=ode_solutions[:, 2], sigma=sigma_Z, observed=Z_observed)


# Phase 3: MCMC Sampling and Analysis
n_draws = 1000
n_tune = 1500
n_chains = 2

with ode_model:
    print("Starting MCMC sampling...")
    idata = pm.sample(
        draws=n_draws,
        tune=n_tune,
        chains=n_chains,
        target_accept=0.9,
        random_seed=RANDOM_SEED,
    )
    print("MCMC sampling completed.")


idata_path = os.path.join(OUTPUT_DIR, "ode_inference_data.nc")
try:
    az.to_netcdf(idata, idata_path)
    print(f"Inference data saved to {idata_path}")
except Exception as e:
    print(f"Error saving inference data: {e}")

print("\nConvergence Diagnostics (Summary):")
summary_vars = param_cols + ['sigma_X', 'sigma_Y', 'sigma_Z']
summary = az.summary(idata, var_names=summary_vars)
print(summary)
summary_path = os.path.join(OUTPUT_DIR, "summary_statistics.csv")
try:
    summary.to_csv(summary_path)
    print(f"Summary statistics saved to {summary_path}")
except Exception as e:
    print(f"Error saving summary statistics: {e}")

print("\nPlotting trace...")
trace_plot_path = os.path.join(OUTPUT_DIR, "trace_plots.png")
try:
    # Dynamically adjust figsize based on the number of variables for better readability
    num_summary_vars = len(summary_vars)
    trace_figsize_height = max(20, num_summary_vars * 1.5) # Adjust multiplier as needed
    az.plot_trace(idata, var_names=summary_vars, figsize=(15, trace_figsize_height))
    plt.tight_layout()
    plt.savefig(trace_plot_path)
    print(f"Trace plots saved to {trace_plot_path}")
    plt.show(block=False)
except Exception as e:
    print(f"Error generating or saving trace plot: {e}")

print("\nPlotting energy plot...")
energy_plot_path = os.path.join(OUTPUT_DIR, "energy_plot.png")
try:
    az.plot_energy(idata)
    plt.tight_layout()
    plt.savefig(energy_plot_path)
    print(f"Energy plot saved to {energy_plot_path}")
    plt.show(block=False)
except Exception as e:
    print(f"Could not generate or save energy plot: {e}")

print("\nPlotting posterior distributions...")
posterior_plot_path = os.path.join(OUTPUT_DIR, "posterior_distributions.png")
ref_val_dict = None
if true_params_series is not None:
    ref_val_dict = {
        param: [{'ref_val': true_params_series[param]}]
        for param in param_cols if param in summary_vars and param in true_params_series
    }
try:
    az.plot_posterior(
        idata,
        var_names=summary_vars,
        ref_val=ref_val_dict if ref_val_dict else None,
        hdi_prob=0.94,
        kind='hist',
        figsize=(15, 12) # Consider making this dynamic too if many vars
    )
    plt.tight_layout()
    plt.savefig(posterior_plot_path)
    print(f"Posterior plots saved to {posterior_plot_path}")
    plt.show(block=False)
except Exception as e:
    print(f"Error generating or saving posterior plot: {e}")

subset_params_for_pairplot = ['r', 'd1', 'c1', 'h1', 'lambda1']
pair_plot_path = os.path.join(OUTPUT_DIR, "pair_plot_subset.png")
if all(p in idata.posterior for p in subset_params_for_pairplot):
    print(f"\nPlotting pair plot for subset: {subset_params_for_pairplot}...")
    try:
        az.plot_pair(
            idata,
            var_names=subset_params_for_pairplot,
            kind='kde',
            marginals=True,
        )
        plt.tight_layout()
        plt.savefig(pair_plot_path)
        print(f"Pair plot saved to {pair_plot_path}")
        plt.show(block=False)
    except Exception as e:
        print(f"Error generating or saving pair plot: {e}")
else:
    print(f"Skipping pair plot as some parameters from {subset_params_for_pairplot} are not in idata.posterior.")

ppc_plot_path = os.path.join(OUTPUT_DIR, "posterior_predictive_checks.png")
print("\nGenerating posterior predictive samples...")
try:
    with ode_model:
        ppc_samples = pm.sample_posterior_predictive(
            idata,
            var_names=['X_obs', 'Y_obs', 'Z_obs'],
            random_seed=RANDOM_SEED,
        )
    print("Posterior predictive sampling completed.")

    print("\nPlotting Posterior Predictive Checks...")
    n_ppc_samples_to_plot = 50
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    time_points = t_observed

    x_ppc_samples = ppc_samples.posterior_predictive["X_obs"]
    for i in range(min(n_ppc_samples_to_plot, x_ppc_samples.sizes["draw"])):
        axes[0].plot(time_points, x_ppc_samples.isel(chain=0, draw=i), c='blue', alpha=0.1)
    axes[0].plot(time_points, X_observed, 'o', color='red', markersize=3, label='Observed X')
    axes[0].plot(time_points, x_ppc_samples.mean(dim=("chain", "draw")), c='black', linestyle='--', label='Mean PPC X')
    axes[0].set_ylabel('X(t)')
    axes[0].legend()
    axes[0].set_title('Posterior Predictive Check for X')

    y_ppc_samples = ppc_samples.posterior_predictive["Y_obs"]
    for i in range(min(n_ppc_samples_to_plot, y_ppc_samples.sizes["draw"])):
        axes[1].plot(time_points, y_ppc_samples.isel(chain=0, draw=i), c='green', alpha=0.1)
    axes[1].plot(time_points, Y_observed, 'o', color='red', markersize=3, label='Observed Y')
    axes[1].plot(time_points, y_ppc_samples.mean(dim=("chain", "draw")), c='black', linestyle='--', label='Mean PPC Y')
    axes[1].set_ylabel('Y(t)')
    axes[1].legend()
    axes[1].set_title('Posterior Predictive Check for Y')

    z_ppc_samples = ppc_samples.posterior_predictive["Z_obs"]
    for i in range(min(n_ppc_samples_to_plot, z_ppc_samples.sizes["draw"])):
        axes[2].plot(time_points, z_ppc_samples.isel(chain=0, draw=i), c='purple', alpha=0.1)
    axes[2].plot(time_points, Z_observed, 'o', color='red', markersize=3, label='Observed Z')
    axes[2].plot(time_points, z_ppc_samples.mean(dim=("chain", "draw")), c='black', linestyle='--', label='Mean PPC Z')
    axes[2].set_ylabel('Z(t)')
    axes[2].set_xlabel('Time')
    axes[2].legend()
    axes[2].set_title('Posterior Predictive Check for Z')

    plt.tight_layout()
    plt.savefig(ppc_plot_path)
    print(f"PPC plot saved to {ppc_plot_path}")
    plt.show(block=False)

except Exception as e:
    print(f"Error during Posterior Predictive Check generation or plotting: {e}")

print("\n--- Script Finished ---")
plt.close('all')