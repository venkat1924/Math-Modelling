import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import itertools
import warnings

# Suppress numerical integration warnings
warnings.filterwarnings('ignore', category=UserWarning)

def ode_system(t, state, params):
    """Enhanced ODE system with numerical safeguards"""
    X, Y, Z = state
    r, p1, p2, d1, b, c1, a, m, h1, d2, d3, lambda1, q = params
    
    # Prevent division by zero in denominators
    denominator1 = max(1 + p1*Y + p2*Z, 1e-12)
    denominator2 = max(a + (1 - m)*X, 1e-12)
    
    # Core dynamics with stabilization
    dXdt = (r * X / denominator1) - d1*X - b*X**2 - (c1*(1 - m)*X*(Y + q*Z)/denominator2)
    dYdt = (h1*(1 - m)*X*(Y + q*Z)/denominator2 - d2*Y - lambda1*Y*Z)
    dZdt = lambda1*Y*Z - (d2 + d3)*Z
    
    # Prevent negative populations and stabilize derivatives
    return [
        max(dXdt, -X/1e-6) if X > 0 else 0,
        max(dYdt, -Y/1e-6) if Y > 0 else 0,
        max(dZdt, -Z/1e-6) if Z > 0 else 0
    ]

def integrate_ode(t_span, y0, params, t_eval=None):
    """Robust integration wrapper with error handling"""
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

    try:
        sol = solve_ivp(ode_system, t_span, y0, args=(params,),
                        t_eval=t_eval, method='LSODA',
                        rtol=1e-6, atol=1e-8,
                        first_step=1e-3, max_step=10)
        
        if sol.success:
            df = pd.DataFrame({
                't': sol.t,
                'X': sol.y[0],
                'Y': sol.y[1],
                'Z': sol.y[2]
            })
            # Post-processing stabilization
            df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].clip(lower=0)
            return df
        return None
    
    except Exception as e:
        print(f"Integration error: {str(e)}")
        return None

def parameter_sweep(parameter_grid, y0, t_span, output_file='simulation_data.csv',
                    t_eval=None, noise_levels=None):
    """Enhanced parameter sweep with diagnostics"""
    param_order = ['r', 'p1', 'p2', 'd1', 'b', 'c1', 'a', 'm',
                   'h1', 'd2', 'd3', 'lambda1', 'q']
    
    param_values = [parameter_grid[param] for param in param_order]
    total_combinations = np.prod([len(v) for v in param_values])
    
    with open(output_file, 'w') as f:
        header = ['run_id', 't', 'X', 'Y', 'Z'] + param_order
        if noise_levels:
            header += ['X_noisy', 'Y_noisy', 'Z_noisy']
        f.write(','.join(header) + '\n')
        
        run_id = 0
        for params in itertools.product(*param_values):
            run_id += 1
            params = list(params)
            
            print(f"Running simulation {run_id}/{total_combinations}", end='\r')
            df = integrate_ode(t_span, y0, params, t_eval)
            
            if df is None:
                print(f"Skipped failed run {run_id}")
                continue
                
            # Add parameters and diagnostics
            df['run_id'] = run_id
            for param_name, param_value in zip(param_order, params):
                df[param_name] = param_value
                
            # Add observational noise
            if noise_levels:
                np.random.seed(run_id)
                for var in ['X', 'Y', 'Z']:
                    clean = df[var].values
                    noise = np.random.normal(0, noise_levels[var], len(clean))
                    df[f'{var}_noisy'] = np.clip(clean * (1 + noise), 0, None)
            
            # Save to CSV
            cols = ['run_id', 't', 'X', 'Y', 'Z'] + param_order
            if noise_levels:
                cols += ['X_noisy', 'Y_noisy', 'Z_noisy']
            
            df[cols].to_csv(f, header=False, index=False, mode='a')
    
    print(f"\nCompleted {run_id} simulations in {output_file}")

if __name__ == "__main__":
    # Configuration
    y0 = [1.0, 0.5, 0.5]  # Initial populations
    t_span = [0, 100]
    t_eval = np.linspace(0, 100, 1000)
    
    # Parameter ranges - adjust these based on your biological system
    parameter_grid = {
        'r': np.linspace(0.1, 2.0, 3),      
        'p1': [0.0, 0.2, 0.4],             
        'p2': [0.1, 0.3],                 
        'd1': [0.05, 0.1],                
        'b': [0.01, 0.05],               
        'c1': [0.1, 0.2],                  
        'a': [0.5, 1.0],                 
        'm': [0.0, 0.1],              
        'h1': [0.2, 0.3],                   
        'd2': [0.05, 0.1],                 
        'd3': [0.02, 0.05],               
        'lambda1': [0.5, 1.0],             
        'q': [0.5, 1.0]                         
    }
    
    # Noise configuration (5% for prey, 10% for predators)
    noise_config = {'X': 0.05, 'Y': 0.1, 'Z': 0.1}
    
    # Run parameter sweep
    parameter_sweep(
        parameter_grid=parameter_grid,
        y0=y0,
        t_span=t_span,
        output_file='simulation_results.csv',
        t_eval=t_eval,
        noise_levels=noise_config
    )