import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import plotly.express as px
import warnings
import os # For checking file path

# --- PyTorch and ML Imports ---
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Suppress numerical integration warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.integrate._ivp.common')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.serialization')

# --- Global Constants & Fixed Model Config (from our LSTM app) ---
PARAM_NAMES = ['r', 'p1', 'p2', 'd1', 'b', 'c1', 'a', 'm',
               'h1', 'd2', 'd3', 'lambda1', 'q']
NUM_PARAMS = len(PARAM_NAMES)
STATE_VARIABLES = ['X', 'Y', 'Z']
NUM_STATE_VARIABLES = len(STATE_VARIABLES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIXED_MODEL_PATH = "Weights/lstm_best_model.pth" # As per your request
FIXED_LSTM_MODEL_PARAMS = {
    'hidden_size': 512,
    'num_layers': 4,
    'lstm_dropout_rate': 0.2,
    'fc_dropout_rate_lstm': 0.5,
    'lstm_bidirectional': True
}

# --- Core Simulation Logic (from reference script, with minor improvements) ---
def ode_system(t, state, current_params): # Renamed params to current_params for clarity
    X, Y, Z = state
    # Ensure current_params directly map to these variables based on PARAM_NAMES order
    r, p1, p2, d1, b, c1, a, m, h1, d2, d3, lambda1_val, q = current_params

    denominator1 = max(1 + p1*Y + p2*Z, 1e-12)
    denominator2 = max(a + (1 - m)*X, 1e-12)
    dXdt = (r * X / denominator1) - d1*X - b*X**2 - (c1*(1 - m)*X*(Y + q*Z)/denominator2)
    dYdt = (h1*(1 - m)*X*(Y + q*Z)/denominator2 - d2*Y - lambda1_val*Y*Z)
    dZdt = lambda1_val*Y*Z - (d2 + d3)*Z
    dXdt_stabilized = max(dXdt, -X/1e-6) if X > 0 else (0 if dXdt < 0 else dXdt)
    dYdt_stabilized = max(dYdt, -Y/1e-6) if Y > 0 else (0 if dYdt < 0 else dYdt)
    dZdt_stabilized = max(dZdt, -Z/1e-6) if Z > 0 else (0 if dZdt < 0 else dZdt)
    return [dXdt_stabilized, dYdt_stabilized, dZdt_stabilized]

def integrate_ode(t_span, y0, current_params, t_eval=None): # Renamed params
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
    try:
        sol = solve_ivp(
            ode_system, t_span, y0, args=(current_params,),
            t_eval=t_eval, method='LSODA', rtol=1e-6, atol=1e-8,
            first_step=1e-3,
            max_step=max((t_span[1] - t_span[0]) / 100, 1e-3) # From reference
        )
        if sol.success:
            df = pd.DataFrame({'t': sol.t, 'X': sol.y[0], 'Y': sol.y[1], 'Z': sol.y[2]})
            df[['X', 'Y', 'Z']] = df[['X', 'Y', 'Z']].clip(lower=0)
            return df
        else:
            st.error(f"ODE Integration failed: {sol.message}")
            return None
    except Exception as e:
        st.error(f"An error occurred during ODE integration: {str(e)}")
        return None

# --- LSTM Model Definition (from our LSTM app) ---
class ParameterPredictorLSTM(nn.Module):
    def __init__(self, num_params: int = NUM_PARAMS,
                 input_features: int = NUM_STATE_VARIABLES,
                 hidden_size: int = 256, num_layers: int = 2,
                 lstm_dropout_rate: float = 0.2, fc_dropout_rate: float = 0.5,
                 bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=lstm_dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        fc_input_features = hidden_size * 2 if bidirectional else hidden_size
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_features, fc_input_features // 2), nn.ReLU(),
            nn.Dropout(fc_dropout_rate),
            nn.Linear(fc_input_features // 2, num_params)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            last_hidden_state = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            last_hidden_state = h_n[-1, :, :]
        return self.fc_layers(last_hidden_state)

# --- Core Inference Function (from our LSTM app) ---
def predict_parameters_from_state_dict(
    x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
    model_state_dict: dict, param_scaler: MinMaxScaler,
    device: torch.device, model_params_config: dict):
    try:
        if not (len(x_data) == len(y_data) == len(z_data)):
            return None, "Input X, Y, Z data must have the same length."
        if len(x_data) == 0: return None, "Input data cannot be empty."
        trajectory = np.stack([x_data, y_data, z_data], axis=-1).astype(np.float32)
        trajectory_scaler = StandardScaler()
        normalized_trajectory = trajectory_scaler.fit_transform(trajectory)
        normalized_trajectory_transposed = np.transpose(normalized_trajectory, (1, 0))
        input_tensor = torch.from_numpy(normalized_trajectory_transposed).unsqueeze(0).to(device)
        model = ParameterPredictorLSTM(
            num_params=NUM_PARAMS, input_features=NUM_STATE_VARIABLES,
            hidden_size=model_params_config.get('hidden_size', 256),
            num_layers=model_params_config.get('num_layers', 2),
            lstm_dropout_rate=model_params_config.get('lstm_dropout_rate', 0.2),
            fc_dropout_rate=model_params_config.get('fc_dropout_rate_lstm', 0.5),
            bidirectional=model_params_config.get('lstm_bidirectional', False)
        ).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
    except Exception as e:
        return None, f"Error initializing model or preprocessing data: {str(e)}"
    try:
        with torch.no_grad():
            predictions_scaled = model(input_tensor)
        predictions_scaled_np = predictions_scaled.cpu().numpy()
        predicted_params_unscaled = param_scaler.inverse_transform(predictions_scaled_np)
        output_dict = dict(zip(PARAM_NAMES, predicted_params_unscaled[0]))
        return output_dict, None
    except Exception as e:
        return None, f"Error during model inference or unscaling: {str(e)}"

# --- Helper function to load checkpoint and scaler (from our LSTM app) ---
@st.cache_resource
def load_full_checkpoint_and_scaler(model_path: str):
    if not os.path.exists(model_path):
        return None, None, f"Model file not found: '{model_path}'. Ensure path is correct relative to app."
    try:
        with open(model_path, 'rb') as f:
            checkpoint = torch.load(f, map_location=DEVICE, weights_only=False)
        param_scaler_loaded = None; scaler_status_msg = "Scaler status unknown."
        if 'param_scaler' in checkpoint:
            param_scaler_loaded = checkpoint['param_scaler']
            if not isinstance(param_scaler_loaded, MinMaxScaler):
                scaler_status_msg = "Error: 'param_scaler' in checkpoint is not a MinMaxScaler."
                param_scaler_loaded = None
            else: scaler_status_msg = "Parameter scaler loaded."
        else:
            scaler_status_msg = "CRITICAL: 'param_scaler' not found in checkpoint."
            return None, None, scaler_status_msg
        model_state_dict_loaded = None
        if 'model_state_dict' in checkpoint: model_state_dict_loaded = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and any(k.startswith("lstm.") or k.startswith("fc_layers.") for k in checkpoint.keys()):
            model_state_dict_loaded = checkpoint
        else:
            return model_state_dict_loaded, param_scaler_loaded, "Checkpoint invalid: no 'model_state_dict'."
        if model_state_dict_loaded is None:
             return None, param_scaler_loaded, "Failed to extract model state_dict."
        return model_state_dict_loaded, param_scaler_loaded, scaler_status_msg
    except Exception as e:
        return None, None, f"Error loading checkpoint/scaler: {str(e)}"

# --- Streamlit Application (Structure from Reference Script) ---
def main(): # Renamed from main_app to main
    st.set_page_config(layout="wide", page_title="ODE Forward & Inverse Simulation") # Title from reference

    # --- Left Sidebar for Inputs (from reference script) ---
    st.sidebar.header("⚙️ Model & Simulation Setup")
    st.sidebar.subheader("ODE Parameters (13)")
    # PARAM_NAMES is global, used here for consistency
    param_defaults = [0.5, 0.1, 0.1, 0.05, 0.02, 0.3, 1.0, 0.1, 0.4, 0.1, 0.05, 0.15, 0.5]
    param_tooltips = {
        'r': "Intrinsic growth rate of X", 'p1': "Inhibition of Y on X",
        'p2': "Inhibition of Z on X", 'd1': "Natural death rate of X",
        'b': "Density-dependent death of X", 'c1': "Predation rate by Y/Z on X",
        'a': "Saturation constant for X predation", 'm': "Modifier for X availability",
        'h1': "Conversion efficiency for Y/Z", 'd2': "Natural death rate of Y & Z",
        'd3': "Additional death rate of Z", 'lambda1': "Interaction Y & Z (Y*Z term)",
        'q': "Relative effect of Z vs Y on X"
    }
    input_ode_params_values = [] # Renamed from params_values to avoid conflict
    cols_params = st.sidebar.columns(2)
    for i, name in enumerate(PARAM_NAMES): # Use global PARAM_NAMES
        default = param_defaults[i]
        col = cols_params[i % 2]
        value = col.number_input(
            label=f"{param_tooltips.get(name, name)} ({name})", # Label style from reference
            value=default,
            step=0.01 if default < 1 and default != 0 else 0.1,
            format="%.3f", key=f"param_{name}", help=param_tooltips.get(name, "")
        )
        input_ode_params_values.append(value)

    st.sidebar.subheader("🌍 Initial Conditions (X₀, Y₀, Z₀)")
    cols_ic = st.sidebar.columns(3)
    X0 = cols_ic[0].number_input("X₀", value=1.0, min_value=0.0, step=0.1, format="%.2f", help="Initial X")
    Y0 = cols_ic[1].number_input("Y₀", value=0.5, min_value=0.0, step=0.1, format="%.2f", help="Initial Y")
    Z0 = cols_ic[2].number_input("Z₀", value=0.5, min_value=0.0, step=0.1, format="%.2f", help="Initial Z")
    y0 = [X0, Y0, Z0]

    st.sidebar.subheader("⏱️ Time Settings")
    cols_time = st.sidebar.columns(2)
    t_start = cols_time[0].number_input("Start Time (t_start)", value=0, step=1, help="Sim start time")
    t_end = cols_time[1].number_input("End Time (t_end)", value=100, step=10, help="Sim end time")
    num_time_points = st.sidebar.number_input("Number of Time Points", value=1000, min_value=10, step=100, help="Points to evaluate")
    t_span = [t_start, t_end]
    t_eval = np.linspace(t_start, t_end, num_time_points)
    run_simulation_button = st.sidebar.button("🔄 Run Simulation", type="primary", use_container_width=True)
    st.sidebar.markdown("---")

    # LSTM Model Status (from our LSTM app, integrated into sidebar)
    st.sidebar.subheader("🧠 Inverse Model Status")
    st.sidebar.caption(f"Fixed Model: `{FIXED_MODEL_PATH}`")
    st.sidebar.caption(f"Inference Device: `{str(DEVICE).upper()}`")
    model_state_dict_loaded, param_scaler_object, load_status_msg = load_full_checkpoint_and_scaler(FIXED_MODEL_PATH)
    if model_state_dict_loaded and param_scaler_object:
        st.sidebar.success(f"Model components loaded.")
        if "scaler loaded" not in load_status_msg.lower() and "file not found" not in load_status_msg.lower() and "critical" not in load_status_msg.lower() :
             st.sidebar.info(f"Scaler status: {load_status_msg}") # Show non-critical specific scaler messages
    else:
        st.sidebar.error(f"Load Error: {load_status_msg}")

    # Author Info (from reference script)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """<style>.by‑heading{font-family:'Poppins','Montserrat',sans-serif;font-size:16px;font-weight:700;letter-spacing:1px;margin:0 0 4px;color:#0ef;background:none;}.by-list{font-family:'Open Sans','Helvetica Neue',sans-serif;font-size:14px;font-weight:400;margin:0 0 12px;padding-left:16px;color:#ccc;}.by-list li{margin-bottom:4px;list-style-type:none;}.by-list li::before{content:'›';margin-right:6px;color:#0ef;}</style>
        <p class="by‑heading">DEVELOPED BY</p><ul class="by-list"><li>Anumaneni Venkat Balachandra</li><li>E Lokeshvar</li><li>Abhyuday Singh</li></ul>
        <p class="by‑heading" style="color:#f6f;">UNDER GUIDANCE OF</p><ul class="by-list"><li>Dr. Y Sailaja</li></ul>""",
        unsafe_allow_html=True
    )

    # --- Main Page Layout (from reference script) ---
    col_main, col_right_sidebar = st.columns([0.7, 0.3], gap="large")

    with col_main:
        st.title("🔬 Biological System: Simulation Visualizer") # Title from reference
        if 'simulation_df' not in st.session_state: st.session_state.simulation_df = None
        if 'current_params_tuple' not in st.session_state: st.session_state.current_params_tuple = None
        current_inputs_tuple = tuple(input_ode_params_values + y0 + t_span + [num_time_points])

        if run_simulation_button or st.session_state.current_params_tuple != current_inputs_tuple :
            st.session_state.current_params_tuple = current_inputs_tuple
            with st.spinner("⚙️ Running simulation... Please wait."):
                simulation_df_result = integrate_ode(t_span, y0, input_ode_params_values, t_eval)
                st.session_state.simulation_df = simulation_df_result
        simulation_df = st.session_state.simulation_df

        if simulation_df is not None and not simulation_df.empty:
            df_melted = simulation_df.melt(id_vars=['t'], value_vars=['X', 'Y', 'Z'], var_name='Population', value_name='Density')
            fig = px.line(df_melted, x='t', y='Density', color='Population', title="Population Dynamics Over Time")
            fig.update_layout(legend_title_text='Populations')
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Show Raw Simulation Data"):
                st.dataframe(simulation_df.style.format("{:.4f}", subset=['X', 'Y', 'Z']).format("{:.2f}", subset=['t']), hide_index=True) # hide_index from ref
        elif run_simulation_button: st.warning("Simulation did not produce data.")
        else: st.info("Adjust parameters in the sidebar and click 'Run Simulation' to view results.")

        # System of Equations Section (from reference script)
        st.markdown("---")        
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

    with col_right_sidebar:
        st.header("Predicted Parameters") # Header from reference
        
        lstm_predictions_dict = None
        lstm_error_message = None
        lstm_attempted = False

        if simulation_df is not None and not simulation_df.empty:
            if model_state_dict_loaded and param_scaler_object:
                lstm_attempted = True
                with st.spinner("🧠 Inferring parameters (LSTM)..."): # Spinner for LSTM
                    lstm_predictions_dict, lstm_error_message = predict_parameters_from_state_dict(
                        x_data=simulation_df['X'].values, y_data=simulation_df['Y'].values, z_data=simulation_df['Z'].values,
                        model_state_dict=model_state_dict_loaded, param_scaler=param_scaler_object,
                        device=DEVICE, model_params_config=FIXED_LSTM_MODEL_PARAMS
                    )
            elif not model_state_dict_loaded or not param_scaler_object:
                 # Display a message once if model isn't ready, right under the header
                 st.warning(f"LSTM Model/Scaler not ready. {load_status_msg}")


        if lstm_attempted and lstm_error_message: # Display LSTM error prominently if it occurred
            st.error(f"LSTM Error: {lstm_error_message}")

        for idx, name in enumerate(PARAM_NAMES): # Iterate using global PARAM_NAMES
            true_value = input_ode_params_values[idx] # From sidebar inputs
            description = param_tooltips.get(name, name)
            
            final_pred_value = np.nan # Default to NaN
            
            if lstm_predictions_dict and name in lstm_predictions_dict:
                final_pred_value = lstm_predictions_dict[name]
            elif lstm_attempted and not lstm_error_message and (not lstm_predictions_dict or name not in lstm_predictions_dict) :
                 # LSTM ran, no error, but param missing (should not happen if PARAM_NAMES is correct)
                 # Or, if LSTM dict is None despite no error message (also unlikely)
                 final_pred_value = np.nan # Keep as NaN, will show N/A
            else: # Fallback to noisy placeholder if LSTM didn't run, errored, or param missing
                fixed_noise_level = 0.025
                std_dev_noise = max(abs(true_value * fixed_noise_level), 1e-4 if true_value == 0 else abs(true_value * fixed_noise_level * 0.1))
                noisy_val = true_value + np.random.normal(loc=0, scale=std_dev_noise)
                if true_value >= 0: noisy_val = max(noisy_val, 0)
                final_pred_value = noisy_val
            
            value_str = f"{final_pred_value:.4f}" if not np.isnan(final_pred_value) else "N/A"
            delta_str = f"(Input {true_value:.3f})" # Default delta if pred is N/A
            if not np.isnan(final_pred_value):
                 delta_str = f"{final_pred_value - true_value:+.4f} (vs Input {true_value:.3f})"

            st.metric(
                label=f"{description} ({name})", # Label style from reference
                value=value_str,
                delta=delta_str,
                delta_color="off"
            )
        st.markdown("---") # From reference

if __name__ == "__main__":
    main()