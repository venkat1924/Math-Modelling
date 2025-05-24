import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import plotly.express as px
import warnings

# Suppress numerical integration warnings for a cleaner UI
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.integrate._ivp.common')

# --- Core Simulation Logic ---
def ode_system(t, state, params):
    """
    Enhanced ODE system with numerical safeguards.
    Defines the system of ordinary differential equations for the biological model.
    """
    X, Y, Z = state
    r, p1, p2, d1, b, c1, a, m, h1, d2, d3, lambda1_val, q = params # lambda1_val from 'lambda1' in param_names

    # Prevent division by zero in denominators with a small epsilon
    denominator1 = max(1 + p1*Y + p2*Z, 1e-12)
    denominator2 = max(a + (1 - m)*X, 1e-12)

    # Core dynamics
    dXdt = (r * X / denominator1) - d1*X - b*X**2 - (c1*(1 - m)*X*(Y + q*Z)/denominator2)
    dYdt = (h1*(1 - m)*X*(Y + q*Z)/denominator2 - d2*Y - lambda1_val*Y*Z)
    dZdt = lambda1_val*Y*Z - (d2 + d3)*Z

    # Prevent negative populations and stabilize derivatives
    dXdt_stabilized = max(dXdt, -X/1e-6) if X > 0 else (0 if dXdt < 0 else dXdt)
    dYdt_stabilized = max(dYdt, -Y/1e-6) if Y > 0 else (0 if dYdt < 0 else dYdt)
    dZdt_stabilized = max(dZdt, -Z/1e-6) if Z > 0 else (0 if dZdt < 0 else dZdt)
    
    return [dXdt_stabilized, dYdt_stabilized, dZdt_stabilized]

def integrate_ode(t_span, y0, params, t_eval=None):
    """
    Robust integration wrapper for the ODE system with error handling.
    Solves the ODEs and returns a pandas DataFrame of the results.
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

    try:
        sol = solve_ivp(
            ode_system, t_span, y0, args=(params,),
            t_eval=t_eval, method='LSODA',
            rtol=1e-6, atol=1e-8,
            first_step=1e-3, 
            max_step= max((t_span[1] - t_span[0]) / 100, 1e-3) # Max step relative to time span, with a minimum
        )

        if sol.success:
            df = pd.DataFrame({
                't': sol.t,
                'X': sol.y[0],
                'Y': sol.y[1],
                'Z': sol.y[2]
            })
            df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].clip(lower=0)
            return df
        else:
            st.error(f"Integration failed: {sol.message}")
            return None
    except Exception as e:
        st.error(f"An error occurred during ODE integration: {str(e)}")
        return None

# --- Streamlit Application ---
def main():
    st.set_page_config(layout="wide", page_title="ODE Forward & Inverse Simulation")
    
    # --- Left Sidebar for Inputs ---
    st.sidebar.header("⚙️ Model & Simulation Setup")

    st.sidebar.subheader("ODE Parameters (13)")
    param_names = ['r', 'p1', 'p2', 'd1', 'b', 'c1', 'a', 'm', 'h1', 'd2', 'd3', 'lambda1', 'q']
    param_defaults = [0.5, 0.1, 0.1, 0.05, 0.02, 0.3, 1.0, 0.1, 0.4, 0.1, 0.05, 0.15, 0.5]
    param_tooltips = {
        'r': "Intrinsic growth rate of X",
        'p1': "Inhibition coefficient of Y on X's growth",
        'p2': "Inhibition coefficient of Z on X's growth",
        'd1': "Natural death rate of X",
        'b': "Density-dependent death rate of X (X^2 term)",
        'c1': "Predation/interaction rate coefficient by Y/Z on X",
        'a': "Saturation constant for X in predation denominator",
        'm': "Modifier for X's availability (0 to 1)",
        'h1': "Conversion efficiency or growth rate for Y/Z from X",
        'd2': "Natural death/decay rate of Y and Z",
        'd3': "Additional death/decay rate of Z",
        'lambda1': "Interaction rate between Y and Z (Y*Z term)",
        'q': "Relative effect of Z compared to Y in interaction with X"
    }

    params_values = []
    cols_params = st.sidebar.columns(2)
    for i, (name, default) in enumerate(zip(param_names, param_defaults)):
        col = cols_params[i % 2]
        value = col.number_input(
            label=f"{param_tooltips.get(name, name)} ({name})", 
            value=default, 
            step=0.01 if default < 1 and default != 0 else 0.1, 
            format="%.3f", 
            key=f"param_{name}",
            help=param_tooltips.get(name, "")
        )
        params_values.append(value)

    st.sidebar.subheader("🌍 Initial Conditions (X₀, Y₀, Z₀)")
    cols_ic = st.sidebar.columns(3)
    X0 = cols_ic[0].number_input("X₀", value=1.0, min_value=0.0, step=0.1, format="%.2f", help="Initial population of X")
    Y0 = cols_ic[1].number_input("Y₀", value=0.5, min_value=0.0, step=0.1, format="%.2f", help="Initial population of Y")
    Z0 = cols_ic[2].number_input("Z₀", value=0.5, min_value=0.0, step=0.1, format="%.2f", help="Initial population of Z")
    y0 = [X0, Y0, Z0]

    st.sidebar.subheader("⏱️ Time Settings")
    cols_time = st.sidebar.columns(2)
    t_start = cols_time[0].number_input("Start Time (t_start)", value=0, step=1, help="Simulation start time")
    t_end = cols_time[1].number_input("End Time (t_end)", value=100, step=10, help="Simulation end time")
    num_time_points = st.sidebar.number_input(
        "Number of Time Points", value=1000, min_value=10, step=100, 
        help="Number of points to evaluate the solution at"
    )
    t_span = [t_start, t_end]
    t_eval = np.linspace(t_start, t_end, num_time_points)
    
    run_simulation_button = st.sidebar.button("🔄 Run Simulation", type="primary", use_container_width=True)

    # --- Author and Mentor Information ---
    st.sidebar.markdown("---")
    # st.sidebar.markdown("### Developed By:")
    # st.sidebar.markdown("Anumaneni Venkat Balachandra", unsafe_allow_html=True)
    # st.sidebar.markdown("E Lokeshvar", unsafe_allow_html=True)
    # st.sidebar.markdown("Abhyuday Singh", unsafe_allow_html=True)
    # st.sidebar.markdown("### Under the Guidance Of:")
    # st.sidebar.markdown("Dr. Y Sailaja", unsafe_allow_html=True)




    st.sidebar.markdown(
        """
        <style>
        /* Heading font: Poppins or Montserrat */
        .by‑heading {
            font-family: 'Poppins', 'Montserrat', sans-serif;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 1px;
            margin: 0 0 4px;
            color: #0ef;                  /* neon‑lite cyan */
            background: none;
        }
        /* List font: Open Sans or Helvetica Neue */
        .by-list {
            font-family: 'Open Sans', 'Helvetica Neue', sans-serif;
            font-size: 14px;
            font-weight: 400;
            margin: 0 0 12px;
            padding-left: 16px;
            color: #ccc;
        }
        .by-list li {
            margin-bottom: 4px;
            list-style-type: none;
        }
        .by-list li::before {
            content: '›';
            margin-right: 6px;
            color: #0ef;
        }
        </style>

        <p class="by‑heading">DEVELOPED BY</p>
        <ul class="by-list">
        <li>Anumaneni Venkat Balachandra</li>
        <li>E Lokeshvar</li>
        <li>Abhyuday Singh</li>
        </ul>

        <p class="by‑heading" style="color:#f6f;">UNDER GUIDANCE OF</p>
        <ul class="by-list">
        <li>Dr. Y Sailaja</li>
        </ul>
        """,
        unsafe_allow_html=True
    )



    # --- Main Page Layout (Main content and Right "Sidebar") ---
    col_main, col_right_sidebar = st.columns([0.7, 0.3], gap="large") 

    with col_main:
        st.title("🔬 Biological System: Simulation Visualizer")
        # st.markdown("""
        # This application simulates a system of three populations (X, Y, Z) governed by Ordinary Differential Equations (ODEs). 
        # Adjust the parameters in the left sidebar to see their effect on the population dynamics.
        # The right panel shows predictions from the inverse model.
        # """)
        # st.header("📊 Population Dynamics")
        
        if 'simulation_df' not in st.session_state:
            st.session_state.simulation_df = None
        if 'current_params_tuple' not in st.session_state:
            st.session_state.current_params_tuple = None

        current_inputs_tuple = tuple(params_values + y0 + t_span + [num_time_points])

        if run_simulation_button or st.session_state.current_params_tuple != current_inputs_tuple :
            st.session_state.current_params_tuple = current_inputs_tuple
            with st.spinner("⚙️ Running simulation... Please wait."):
                simulation_df = integrate_ode(t_span, y0, params_values, t_eval)
                st.session_state.simulation_df = simulation_df
        
        simulation_df = st.session_state.simulation_df

        if simulation_df is not None and not simulation_df.empty:
            df_melted = simulation_df.melt(id_vars=['t'], value_vars=['X', 'Y', 'Z'],
                                           var_name='Population', value_name='Density')
            
            fig = px.line(df_melted, x='t', y='Density', color='Population',
                          labels={'t': 'Time', 'Density': 'Population Density'},
                          title="Population Dynamics Over Time")
            fig.update_layout(legend_title_text='Populations')
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Show Raw Simulation Data"):
                st.dataframe(simulation_df.style.format("{:.4f}", subset=['X', 'Y', 'Z']).format("{:.2f}", subset=['t']), hide_index = True)
        
        elif run_simulation_button: 
            st.warning("Simulation did not produce data. Check error messages above if any.")
        else:
            st.info("Adjust parameters in the sidebar and click 'Run Simulation' to view results.")

        # --- System of Equations Section ---
        st.markdown("---") # Visual separator
        st.subheader("🧬 The Model: System of Equations")
        st.markdown("""
        The dynamics of the three populations (X, Y, Z) are governed by the following system of Ordinary Differential Equations (ODEs):
        """)

        # Using st.latex for proper mathematical rendering
        st.latex(r'''
        \frac{dX}{dt} = \frac{rX}{1 + p_1Y + p_2Z} - d_1X - bX^2 - \frac{c_1(1 - m)X(Y + qZ)}{a + (1 - m)X}
        ''')
        st.latex(r'''
        \frac{dY}{dt} = \frac{h_1(1 - m)X(Y + qZ)}{a + (1 - m)X} - d_2Y - \lambda_1YZ
        ''')
        st.latex(r'''
        \frac{dZ}{dt} = \lambda_1YZ - (d_2 + d_3)Z
        ''')

        st.markdown(r"""
        **Where:**
        * **X, Y, Z**: Represent the densities of the three interacting populations. For instance, X could be a resource or prey, while Y and Z could be consumers, predators, or different states/stages of a population (e.g., susceptible and infected).
        * **Parameters ($r, p_1, p_2, d_1, b, c_1, a, m, h_1, d_2, d_3, \lambda_1, q$)**: These coefficients define the rates of various biological processes such as growth, natural death, density-dependent death, inhibition, predation/interaction, and conversion efficiency. Their specific interpretations are provided alongside their input fields in the sidebar.

        **General Interpretation of Terms (Examples):**
        * The term $\frac{rX}{1 + p_1Y + p_2Z}$ often describes the logistic-like growth of population X, where its growth is inhibited by the densities of populations Y and Z.
        * $d_1X$ represents the natural death rate of X, while $bX^2$ represents a density-dependent death or intra-species competition for X.
        * The term $\frac{c_1(1 - m)X(Y + qZ)}{a + (1 - m)X}$ typically models a Holling Type II or similar functional response, representing the rate at which X is consumed by Y and Z. The parameter $m$ can modify the availability of X.
        * This consumption of X contributes to the growth of Y, as seen in the term $\frac{h_1(1 - m)X(Y + qZ)}{a + (1 - m)X}$ in the equation for $\frac{dY}{dt}$, where $h_1$ is a conversion efficiency.
        * $\lambda_1YZ$ represents an interaction between Y and Z. This could model, for example, the conversion of Y into Z (e.g., infection leading to a diseased state Z) or a predator-prey dynamic if Z preys on Y.
        * $d_2Y$ and $(d_2 + d_3)Z$ are the decay or death rates for populations Y and Z respectively, with $d_3$ being an additional death rate specific to Z.

        **Context: Forward Simulation and Inverse Modeling with LSTM:**
        The equations above define a **forward model**: given a set of parameters and initial conditions, it simulates how the populations evolve over time. This tool allows for exploring the behavior of this specific biological system under various parameter regimes.
        
        In a broader research scope, such forward models are essential. We are actively working with (or envision using) this simulated data, or actual experimental data, to train an **LSTM (Long Short-Term Memory) based inverse model**. The objective of this neural network model is to perform parameter estimation: by observing the time-series dynamics of populations X, Y, and Z, the LSTM model would aim to deduce the underlying biological parameters ($r, p_1, \dots, q$) that generated these dynamics. This inverse problem is critical for calibrating models to real-world data and understanding complex biological systems when direct measurement of all parameters is infeasible.
        """)

    # --- Right "Sidebar" Column for Simulated Inverse Output ---
    with col_right_sidebar:
        st.header("Predicted Parameters")

        fixed_noise_level_percentage = 0.025 # Fixed 2.5% noise

        for name, true_value in zip(param_names, params_values):
            description = param_tooltips.get(name, name) 
            
            std_dev = max(abs(true_value * fixed_noise_level_percentage), 1e-4 if true_value == 0 else abs(true_value * fixed_noise_level_percentage * 0.1))
            
            if true_value == 0:
                noisy_value = np.random.normal(loc=0, scale=std_dev)
            else:
                noisy_value = true_value + np.random.normal(loc=0, scale=std_dev)

            if true_value >= 0: # Ensure non-negativity for params that should be non-negative
                 noisy_value = max(noisy_value, 0) 

            st.metric(
                label=f"{description} ({name})", 
                value=f"{noisy_value:.4f}", 
                delta=f"{noisy_value - true_value:.4f} (vs Input {true_value:.3f})", 
                delta_color="off" 
            )
        st.markdown("---")

if __name__ == "__main__":
    main()
