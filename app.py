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
    elo_a = st.number_input("Elo Generico A", min_value=1000, max_value=2500, value=1500, step=1, key="elo_a")
    q_a = st.number_input("Quota Book A", 1.01, 10.0, 1.64, step=0.01, key="q_a")
    q_pin_a = st.number_input("Quota Pinnacle A", 1.01, 10.0, 1.68, step=0.01, key="q_pin_a")

with col2:
    st.subheader("Giocatore B")
    text_b = st.text_area("Incolla qui la riga delle statistiche (es. Clay ...)", height=100, key="text_b")
    elo_b = st.number_input("Elo Generico B", min_value=1000, max_value=2500, value=1500, step=1, key="elo_b")
    q_b = st.number_input("Quota Book B", 1.01, 10.0, 2.12, step=0.01, key="q_b")
    q_pin_b = st.number_input("Quota Pinnacle B", 1.01, 10.0, 2.27, step=0.01, key="q_pin_b")

if st.button("🚀 Calcola Edge", type="primary"):
    if text_a and text_b:
        data_a = parse_ta_stats(text_a)
        data_b = parse_ta_stats(text_b)
        
        if data_a and data_b:
            st.info(f"✅ **Dati A letti**: 1stIn: {data_a['1stIn']:.1%} | 1stW: {data_a['1stW']:.1%} | 2ndW: {data_a['2ndW']:.1%} | RPW: {data_a['RPW']:.1%}")
            st.info(f"✅ **Dati B letti**: 1stIn: {data_b['1stIn']:.1%} | 1stW: {data_b['1stW']:.1%} | 2ndW: {data_b['2ndW']:.1%} | RPW: {data_b['RPW']:.1%}")
            
            # 1. Calcolo Markoviano (basato sulle stats incollate)
            p_point_a = calc_p_point(data_a['1stIn'], data_a['1stW'], data_a['2ndW'], data_a['ret_first'], data_a['ret_second'])
            p_hold_a = hold_probability(p_point_a)
            
            p_point_b = calc_p_point(data_b['1stIn'], data_b['1stW'], data_b['2ndW'], data_b['ret_first'], data_b['ret_second'])
            p_hold_b = hold_probability(p_point_b)
            
            p_game_A = (p_hold_a + (1 - p_hold_b)) / 2
            p_markov_A = p_game_A**2 * (3 - 2 * p_game_A)
            
            # 2. Calcolo probabilità basata sull'Elo (Formula standard)
            elo_diff = elo_a - elo_b
            p_elo_A = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            
            # 3. Fusione Ibrida: 60% Statistiche di gioco, 40% Forza reale (Elo)
            p_final_A = (p_markov_A * 0.60) + (p_elo_A * 0.40)
            
            # 4. Calcolo Edge
            margine_book = (1/q_a) + (1/q_b)
            prob_fair_book_a = (1/q_a) / margine_book
            edge_book = p_final_A - prob_fair_book_a
            
            margine_pin = (1/q_pin_a) + (1/q_pin_b)
            prob_fair_pin_a = (1/q_pin_a) / margine_pin
            edge_pin = p_final_A - prob_fair_pin_a
            
            # Visualizzazione
            st.divider()
            st.subheader(" Risultati Modello Ibrido")
            
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Prob. Finale A", f"{p_final_A:.1%}", f"(Stats: {p_markov_A:.1%} | Elo: {p_elo_A:.1%})")
            res_col2.metric("Quota Fair", f"@{1/p_final_A:.2f}")
            res_col3.metric("Edge vs TUO BOOK", f"{edge_book*100:+.1f}%")
            res_col4.metric("Edge vs PINNACLE", f"{edge_pin*100:+.1f}%")
            
            if edge_book > 0.05 and abs(edge_pin) < 0.08:
                st.success("🔥 VALORE REALE: Il modello e Pinnacle sono allineati, il tuo book è in ritardo. Scommetti.")
            elif edge_book > 0.05 and abs(edge_pin) >= 0.08:
                st.warning("⚠️ TRAPPOLA DI VALORE: Il tuo book offre una quota alta, ma Pinnacle (e l'Elo) sanno che è rischioso. Evita.")
            else:
                st.error("❌ NESSUN VALORE: La quota non offre edge matematico.")
        else:
            st.error("Impossibile leggere le statistiche. Assicurati di copiare l'intera riga 'Clay' dalla tabella Splits di Tennis Abstract.")
    else:
        st.warning("Incolla i dati e inserisci l'Elo di entrambi i giocatori.")

st.divider()
st.caption("Modello Ibrido: 60% Markov (Serve/Return) + 40% Elo (Forza reale).")
