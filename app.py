import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Tennis Quant Model", layout="wide")

# --- 1. FUNZIONI MATEMATICHE (MOTORE MARKOV) ---
def logit(x, eps=1e-6):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))

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

# --- 2. INTERFACCIA STREAMLIT ---
st.title("🎾 Tennis Quant: Google Sheets Edition")
st.markdown("Modello probabilistico Markoviano per l'identificazione di valore nelle quote.")

st.divider()

# SEZIONE LETTORE GOOGLE SHEETS (Link già inserito)
url_sheet = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTYgWgfwyK8mNBgoG8LEUALbEwjHH7UOCrapR4JNgbi5C1h0xT10i4eICW7FqU6R32EzFwIwyLjFMrl/pub?output=csv"

if url_sheet:
    try:
        df = pd.read_csv(url_sheet)
        
        required_cols = ['A_first_in', 'A_first_w', 'A_second_w', 'A_ret_first', 'A_ret_second', 
                         'B_first_in', 'B_first_w', 'B_second_w', 'B_ret_first', 'B_ret_second', 'Quota_A', 'Quota_B']
        
        if not all(col in df.columns for col in required_cols):
            st.error("Errore: Il foglio non ha le colonne corrette. Controlla le intestazioni.")
        else:
            st.success(f"✅ Foglio caricato! Trovate {len(df)} partite.")
            
            results = []
            for index, row in df.iterrows():
                p_point_a = calc_p_point(row['A_first_in'], row['A_first_w'], row['A_second_w'], row['A_ret_first'], row['A_ret_second'])
                p_hold_a = hold_probability(p_point_a)
                
                p_point_b = calc_p_point(row['B_first_in'], row['B_first_w'], row['B_second_w'], row['B_ret_first'], row['B_ret_second'])
                p_hold_b = hold_probability(p_point_b)
                
                   # Probabilità media che A vinca un game (servizio proprio + risposta)
p_game_A = (p_hold_a + (1 - p_hold_b)) / 2
# Match best-of-3 (approssimazione)
p_match_a = p_game_A**2 * (3 - 2*p_game_A) 
                
                q_a = float(row['Quota_A'])
                q_b = float(row['Quota_B'])
                margine = (1/q_a) + (1/q_b)
                prob_fair_a = (1/q_a) / margine
                edge_a = p_match_a - prob_fair_a
                
                results.append({
                    'Partita': "Giocatore A vs B",
                    'P(Hold) A': f"{p_hold_a:.1%}",
                    'P(Hold) B': f"{p_hold_b:.1%}",
                    'P(Match) A': f"{p_match_a:.1%}",
                    'Quota Fair A': f"@{1/p_match_a:.3f}",
                    'Quota Book A': f"@{q_a}",
                    'EDGE %': f"{edge_a:.1%}",
                    'Verdetto': "✅ VALUE" if edge_a > 0 else "❌ NO"
                })
            
            st.subheader("📊 Analisi Partite")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            
            value_bets = [r for r in results if "VALUE" in r['Verdetto']]
            if value_bets:
                st.subheader("🔥 Value Bet Trovate")
                st.dataframe(pd.DataFrame(value_bets), use_container_width=True)
                
    except Exception as e:
        st.error(f"Errore nel caricamento del foglio: {e}")

st.divider()
st.caption("Sviluppato con modello Markoviano punto-per-punto.")
