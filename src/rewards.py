import numpy as np

def calculate_mec_reward(metrics, intent_vector, context):
    if not metrics['deadline_met']:
        return -2.0 # Harsh penalty for unity
        
    try:
        # Debug
        # print(f"DEBUG: metrics={metrics.keys()} intent={intent_vector}")
    
        # --- Energy Physics (Linear Battery Multiplier) ---
        # User Request: battery_multiplier = 1.0 + ((100.0 - battery) / 25.0)
        battery_j = context.get('ue_battery_joules', 1000.0) # Default full
        battery_pct = (battery_j / 1000.0) * 100.0
        battery_multiplier = 1.0 + ((100.0 - battery_pct) / 25.0)
        
        # Normalization
        thr_bps = context.get('serving_throughput_bps', 0.0)
        MAX_THR = 1e9 
        MAX_LAT = 0.3 # Hardened
        MAX_ENG = 1.0 
        
        g_thr = min(thr_bps / MAX_THR, 1.0)
        lat_val = metrics.get('latency_s', 0.3)
        g_lat = 1.0 - min(lat_val / MAX_LAT, 1.0)
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
    # Normalized SINR (Goodness)
    sinr = context['serving_sinr_db']
    g_thr = np.clip((sinr + 10.0) / 40.0, 0.0, 1.0)
    
    # Stability
    is_ho = (context['serving_cell_id'] != prev_context['serving_cell_id'])
    g_lat = 0.0 if is_ho else 1.0
    
    # Coverage
    rsrp = context['serving_rsrp_dbm']
    g_eng = np.clip((rsrp + 120.0) / 70.0, 0.0, 1.0)
    
    w_thr, w_lat, w_eng = intent_vector
    weighted_score = (w_thr * g_thr) + (w_lat * g_lat) + (w_eng * g_eng)
    
    # Penalty for HO
    penalty = 0.0
    if is_ho:
        penalty = (2.0 * w_lat) + 0.1
        
    return weighted_score - penalty
