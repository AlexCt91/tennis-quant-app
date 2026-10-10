import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Tennis Quant Pro", layout="wide")

# --- 1. FUNZIONI MATEMATICHE ---
def logit(x, eps=1e-6):
    return np.log(np.clip(x, eps, 1.0 - eps) / (1.0 - np.clip(x, eps, 1.0 - eps)))

def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

def combine_rates(server_rate, opp_return_rate, weight=0.5, eps=1e-6):
    server_rate = np.clip(server_rate, eps, 1.0 - eps)
    implied_from_return = np.clip(1.0 - opp_return_rate, eps, 1.0 - eps)
    z = weight * logit(server_rate, eps) + (1 - weight) * logit(implied_from_return, eps)
    return inv_logit(z)

def hold_probability(p, eps=1e-9):
    p = float(np.clip(p, eps, 1.0 - eps))
    q = 1.0 - p
    win_before_deuce = p**4 * (1.0 + 4.0 * q + 10.0 * q**2)
    reach_deuce = 20.0 * p**3 * q**3
    win_from_deuce = (p**2) / (p**2 + q**2)
    return float(win_before_deuce + reach_deuce * win_from_deuce)

def calc_p_point(first_in, first_w, second_w, ret_first, ret_second):
    p1 = combine_rates(first_w, ret_first, 0.5)
    p2 = combine_rates(second_w, ret_second, 0.5)
    return first_in * p1 + (1.0 - first_in) * p2

def parse_ta_stats(text):
    """Estrae le statistiche dalla riga copiata da Tennis Abstract"""
    tokens = text.split()
    surface_idx = -1
    for s in ['Clay', 'Hard', 'Grass', 'Carpet', 'Total', 'All']:
        if s in tokens:
            surface_idx = tokens.index(s)
            break
            
    if surface_idx == -1 or len(tokens) < surface_idx + 20:
        return None
        
    def get_val(idx):
        val_str = tokens[surface_idx + idx]
        val_str = val_str.replace('%', '').replace(',', '.')
        try:
            return float(val_str) / 100.0 if '%' in tokens[surface_idx + idx] else float(val_str)
        except:
            return 0.0

    first_in = get_val(16)
    first_w = get_val(17)
    second_w = get_val(18)
    rpw = get_val(20)
    
    ret_first = max(0.0, rpw - 0.05)
    ret_second = min(1.0, rpw + 0.05)
    
    return {
        '1stIn': first_in, '1stW': first_w, '2ndW': second_w,
        'ret_first': ret_first, 'ret_second': ret_second, 'RPW': rpw
    }

# --- 2. INTERFACCIA ---
st.title("🎾 Tennis Quant Pro: Stats + Elo")
st.markdown("1. Copia la riga 'Clay' da Tennis Abstract. 2. Inserisci l'Elo generico. 3. Calcola.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Giocatore A")
    text_a = st.text_area("Incolla qui la riga delle statistiche (es. Clay ...)", height=100, key="text_a")
    elo_a = st.number_input("Elo Generico A", min_value=
