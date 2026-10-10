import streamlit as st
import pandas as pd
import numpy as np
import io

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

def parse_ta_text(text):
    """Estrae i dati da una riga copiata da Tennis Abstract"""
    try:
        # Legge il testo copiato come una tabella separata da tabulazioni
        df = pd.read_csv(io.StringIO(text), sep='\t', header=None)
        row = df.iloc[0].astype(str)
        
        # Mappatura approssimativa delle colonne di Tennis Abstract (Last 52 Weeks o Career)
        # Di solito: Surface, W, L, Win%, Set W-L, Set%, Game W-L, Game%, TB W-L, TB%, MS, Hld%, Brk%, A%, DF%, 1stIn, 1st%, 2nd%, SPW, RPW, TPW, DR
        # Adattiamo gli indici in base alla struttura standard di TA
        data = {}
        # Cerchiamo di trovare i valori numerici. Questa è una logica robusta di fallback
        nums = []
        for val in row:
            try:
                # Pulisce le percentuali (es. "61.0%" -> 0.61)
                clean_val = val.replace('%', '').replace(',', '.')
                nums.append(float(clean_val) / 100.0 if '%' in val else float(clean_val))
            except:
                nums.append(np.nan)
        
        # Assumiamo che l'utente copi la riga "Clay" o "Hard". 
        # Gli indici tipici in TA sono: 14=1stIn, 15=1st%, 16=2nd%, 18=RPW
        # Se la riga è più corta o più lunga, questo cerca di adattarsi.
        data['1stIn'] = nums[14] if len(nums) > 14 else 0.60
        data['1stW'] = nums[15] if len(nums) > 15 else 0.70
        data['2ndW'] = nums[16] if len(nums) > 16 else 0.50
        data['RPW'] = nums[18] if len(nums) > 18 else 0.40
        
        # Stima ret_first e ret_second da RPW se non disponibili singolarmente
        data['ret_first'] = data['RPW'] - 0.05
        data['ret_second'] = data['RPW'] + 0.05
        
        # Tentativo di estrarre ELO se presente nel testo (cerca un numero a 4 cifre)
        import re
        elo_match = re.search(r'\b(1[0-9]{3}|2[0-9]{3})\b', text)
        data['ELO'] = int(elo_match.group()) if elo_match else 1500
        
        return data
    except Exception as e:
        return None

# --- 2. INTERFACCIA ---
st.title("🎾 Tennis Quant Pro: Stats + ELO")
st.markdown("Copia la riga da Tennis Abstract e incollala qui. Il modello fonde Stats e ELO.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Giocatore A")
    text_a = st.text_area("Incolla qui la riga da Tennis Abstract (Giocatore A)", height=100, placeholder="Es: Clay 114 61 53 53.5% ...")
    q_a = st.number_input("Quota Book A", 1.01, 10.0, 2.00, step=0.01)
    q_pin_a = st.number_input("Quota Pinnacle A", 1.01, 10.0, 2.00, step=0.01)

with col2:
    st.subheader("Giocatore B")
    text_b = st.text_area("Incolla qui la riga da Tennis Abstract (Giocatore B)", height=100, placeholder="Es: Clay 114 61 53 53.5% ...")
    q_b = st.number_input("Quota Book B", 1.01, 10.0, 1.80, step=0.01)
    q_pin_b = st.number_input("Quota Pinnacle B", 1.01, 10.0, 1.80, step=0.01)

if st.button("🚀 Calcola Edge", type="primary"):
    if text_a and text_b:
        data_a = parse_ta_text(text_a)
        data_b = parse_ta_text(text_b)
        
        if data_a and data_b:
            # 1. Calcolo Markoviano
            p_point_a = calc_p_point(data_a['1stIn'], data_a['1stW'], data_a['2ndW'], data_a['ret_first'], data_a['ret_second'])
            p_hold_a = hold_probability(p_point_a)
            
            p_point_b = calc_p_point(data_b['1stIn'], data_b['1stW'], data_b['2ndW'], data_b['ret_first'], data_b['ret_second'])
            p_hold_b = hold_probability(p_point_b)
            
            p_game_A = (p_hold_a + (1 - p_hold_b)) / 2
            p_markov_A = p_game_A**2 * (3 - 2 * p_game_A)
            
            # 2. Calcolo ELO (Formula standard)
            # Differenza ELO: se A ha 1600 e B ha 1500, diff = 100
            elo_diff = data_a['ELO'] - data_b['ELO']
            p_elo_A = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            
            # 3. Fusione Ibrida (60% Stats, 40% ELO)
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
            st.subheader("📊 Risultati Modello Ibrido")
            
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Prob. Finale A", f"{p_final_A:.1%}", f"(Stats: {p_markov_A:.1%} | ELO: {p_elo_A:.1%})")
            res_col2.metric("Quota Fair", f"@{1/p_final_A:.2f}")
            res_col3.metric("Edge vs TUO BOOK", f"{edge_book*100:+.1f}%")
            res_col4.metric("Edge vs PINNACLE", f"{edge_pin*100:+.1f}%")
            
            if edge_book > 0.05 and abs(edge_pin) < 0.08:
                st.success("🔥 VALORE REALE: Il modello e Pinnacle sono allineati, il tuo book è in ritardo. Scommetti.")
            elif edge_book > 0.05 and abs(edge_pin) >= 0.08:
                st.warning("⚠️ TRAPPOLA DI VALORE: Il tuo book offre una quota alta, ma Pinnacle (e il nostro ELO) sanno che è rischioso. Evita.")
            else:
                st.error("❌ NESSUN VALORE: La quota non offre edge matematico.")
        else:
            st.error("Impossibile leggere i dati incollati. Assicurati di copiare l'intera riga della tabella da Tennis Abstract.")
    else:
        st.warning("Incolla i dati di entrambi i giocatori.")

st.divider()
st.caption("Modello Ibrido: 60% Markov (Serve/Return) + 40% ELO (Forza avversario).")
