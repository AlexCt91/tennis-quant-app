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
st.title("🎾 Tennis Quant: Sharp vs Soft Bookmaker")
st.markdown("Modello Markoviano + Confronto Quote Pinnacle (Sharp) vs Tuo Book (Soft).")
st.divider()

url_sheet = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTYgWgfwyK8mNBgoG8LEUALbEwjHH7UOCrapR4JNgbi5C1h0xT10i4eICW7FqU6R32EzFwIwyLjFMrl/pub?output=csv"

if url_sheet:
    try:
        df = pd.read_csv(url_sheet)
        
        # Colonne richieste (aggiunte Pinnacle)
        required_cols = ['A_first_in', 'A_first_w', 'A_second_w', 'A_ret_first', 'A_ret_second', 
                         'B_first_in', 'B_first_w', 'B_second_w', 'B_ret_first', 'B_ret_second', 
                         'Quota_A', 'Quota_B', 'Quota_Pin_A', 'Quota_Pin_B']
        
        if not all(col in df.columns for col in required_cols):
            st.error("Errore: Il foglio non ha le colonne corrette. Assicurati di aver aggiunto 'Quota_Pin_A' e 'Quota_Pin_B'.")
        else:
            st.success(f"✅ Foglio caricato! Trovate {len(df)} partite.")
            
            results = []
            for index, row in df.iterrows():
                p_point_a = calc_p_point(row['A_first_in'], row['A_first_w'], row['A_second_w'], row['A_ret_first'], row['A_ret_second'])
                p_hold_a = hold_probability(p_point_a)
                
                p_point_b = calc_p_point(row['B_first_in'], row['B_first_w'], row['B_second_w'], row['B_ret_first'], row['B_ret_second'])
                p_hold_b = hold_probability(p_point_b)
                
                p_game_A = (p_hold_a + (1 - p_hold_b)) / 2
                p_match_a = p_game_A**2 * (3 - 2 * p_game_A)
                
                q_a = float(row['Quota_A'])
                q_b = float(row['Quota_B'])
                margine_book = (1/q_a) + (1/q_b)
                prob_fair_book_a = (1/q_a) / margine_book
                edge_book = p_match_a - prob_fair_book_a
                
                # Calcolo Pinnacle (se presente)
                q_pin_a = float(row['Quota_Pin_A']) if pd.notna(row['Quota_Pin_A']) else None
                q_pin_b = float(row['Quota_Pin_B']) if pd.notna(row['Quota_Pin_B']) else None
                
                edge_pin = "N/D"
                if q_pin_a and q_pin_b:
                    margine_pin = (1/q_pin_a) + (1/q_pin_b)
                    prob_fair_pin_a = (1/q_pin_a) / margine_pin
                    edge_pin = f"{(p_match_a - prob_fair_pin_a)*100:.1f}%"
                
                results.append({
                    'P(Match) A': f"{p_match_a:.1%}",
                    'Quota Fair': f"@{1/p_match_a:.2f}",
                    'Quota PIN': f"@{q_pin_a}" if q_pin_a else "N/D",
                    'Edge vs PIN': edge_pin,
                    'Quota TUO BOOK': f"@{q_a}",
                    'Edge vs TUO BOOK': f"{edge_book*100:.1f}%",
                    'Verdetto': "✅ Scommetti" if edge_book > 0.05 else "❌ Passa"
                })
            
            st.subheader("📊 Analisi Comparativa")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
                
    except Exception as e:
        st.error(f"Errore nel caricamento del foglio: {e}")

st.divider()
st.caption("Sviluppato con modello Markoviano punto-per-punto.")
