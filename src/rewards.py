import numpy as np

def calculate_mec_reward(metrics, intent_vector, context):
    if not metrics['deadline_met']:
        return -2.0 # Harsh penalty for unity
        
    try:
        # Debug
        # print(f"DEBUG: metrics={metrics.keys()} intent={intent_vector}")
    
        # --- Energy Physics (Linear Battery Multiplier) ---
        # battery_multiplier = 1.0 + ((100.0 - battery) / 25.0)
        max_battery = context.get('ue_max_battery_joules', 1000.0) 
        battery_j = context.get('ue_battery_joules', 1000.0) # Default full
        battery_pct = (battery_j / max_battery) * 100.0
        battery_multiplier = 1.0 + ((100.0 - battery_pct) / 50.0)
        
        # Non-Linear Utility Normalization
        thr_bps = context.get('serving_throughput_bps', 0.0)
        # Tanh Utility: Maps typical range (50-100Mbps) to steep part of curve
        g_thr = np.tanh(thr_bps / 100e6)
        
        lat_val = metrics.get('latency_s', 0.3)
        deadline = metrics.get('deadline_s', 0.3)
        # Exponential Decay Utility: Rewards being closer to zero latency
        g_lat = np.exp(-lat_val / max(deadline, 1e-9))
        
        eng_val = metrics.get('energy_j', 1.0)
        
        # --- Energy Score Logic ---
        w_thr, w_lat, w_eng = intent_vector
    
        # New Logic: penalty = energy * beta * battery_multiplier
        penalty = eng_val * w_eng * battery_multiplier
        
        # Weighted Score (Latency + Throughput only)
        # Energy is handled strictly via penalty
        weighted_score = (w_thr * g_thr) + (w_lat * g_lat)
        
        # Apply Physics Penalty
        return 1.0 + weighted_score - penalty

    except Exception as e:
        print(f"CRASH IN REWARD: {e}")
        print(f"Context keys: {context.keys()}")
        raise e

def calculate_ho_reward(context, prev_context, intent_vector):
    # --- Smooth Utility Normalization (Sigmoid S-Curves) ---
    # SINR Goodness: Centered at 10 dB, Scale: 5.0 (Covers 0-20dB spectral efficiency zone)
    sinr = context.get('serving_sinr_db', 0.0)
    g_thr = 1.0 / (1.0 + np.exp(-(sinr - 10.0) / 5.0))
    
    # Stability: 1.0 if we stayed in the same cell, 0.0 if we handed over
    is_ho = (context['serving_cell_id'] != prev_context['serving_cell_id'])
    g_lat = 0.0 if is_ho else 1.0
    
    # Coverage/Signal: Centered at -95 dBm, Scale: 10.0 (Provides safety buffer above RLF)
    rsrp = context.get('serving_rsrp_dbm', -100.0)
    g_eng = 1.0 / (1.0 + np.exp(-(rsrp + 95.0) / 10.0))
    
    w_thr, w_lat, w_eng = intent_vector
    weighted_score = (w_thr * g_thr) + (w_lat * g_lat) + (w_eng * g_eng)
    
    # Penalty for HO
    penalty = 0.0
    if is_ho:
        penalty = (2.0 * w_lat) + 0.1
        
    return weighted_score - penalty
