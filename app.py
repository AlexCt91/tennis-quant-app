import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Tennis Quant Pro", layout="wide")

# --- 1. FUNZIONI MATEMATICHE (Condivise) ---
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
    tokens = text.split()
    surface_idx = -1
    for s in ['Clay', 'Hard', 'Grass', 'Carpet', 'Total', 'All']:
        if s in tokens:
            surface_idx = tokens.index(s)
            break
    if surface_idx == -1 or len(tokens) < surface_idx + 20:
        return None
    def get_val(idx):
        val_str = tokens[surface_idx + idx].replace('%', '').replace(',', '.')
        try:
            return float(val_str) / 100.0 if '%' in tokens[surface_idx + idx] else float(val_str)
        except:
            return 0.0
    return {
        '1stIn': get_val(16), '1stW': get_val(17), '2ndW': get_val(18),
        'ret_first': max(0.0, get_val(20) - 0.05), 'ret_second': min(1.0, get_val(20) + 0.05)
    }

# --- 2. INTERFACCIA CON SCHEDE ---
st.title("🎾 Tennis Quant Pro: Suite Completa")
st.markdown("Scegli la modalità di analisi:")

tab1, tab2 = st.tabs(["🎯 Analisi Singola (Manuale + Elo)", "📑 Analisi Massiva (Google Sheets)"])

# ================= TAB 1: MANUALE =================
with tab1:
    st.subheader("Analisi Profonda Singola Partita")
    st.markdown("Ideale per 1-2 match. Copia la riga 'Clay' da Tennis Abstract e inserisci l'Elo reale.")
    col1, col2 = st.columns(2)
    with col1:
        text_a = st.text_area("Stats Giocatore A", height=100, key="t1_text_a")
        elo_a = st.number_input("Elo A (es. 1559)", min_value=1000, max_value=2500, value=1500, step=1, key="t1_elo_a")
        q_a = st.number_input("Quota Book A", 1.01, 10.0, 1.66, step=0.01, key="t1_q_a")
        q_pin_a = st.number_input("Quota Pinnacle A", 1.01, 10.0, 1.68, step=0.01, key="t1_q_pin_a")
    with col2:
        text_b = st.text_area("Stats Giocatore B", height=100, key="t1_text_b")
        elo_b = st.number_input("Elo B (es. 1492)", min_value=1000, max_value=2500, value=1500, step=1, key="t1_elo_b")
        q_b = st.number_input("Quota Book B", 1.01, 10.0, 2.09, step=0.01, key="t1_q_b")
        q_pin_b = st.number_input("Quota Pinnacle B", 1.01, 10.0, 2.27, step=0.01, key="t1_q_pin_b")

    if st.button("🚀 Calcola Edge (Singolo)", type="primary", key="btn_1"):
        if text_a and text_b:
            data_a, data_b = parse_ta_stats(text_a), parse_ta_stats(text_b)
            if data_a and data_b:
                p_point_a = calc_p_point(data_a['1stIn'], data_a['1stW'], data_a['2ndW'], data_a['ret_first'], data_a['ret_second'])
                p_hold_a = hold_probability(p_point_a)
                p_point_b = calc_p_point(data_b['1stIn'], data_b['1stW'], data_b['2ndW'], data_b['ret_first'], data_b['ret_second'])
                p_hold_b = hold_probability(p_point_b)
                
                p_game_A = (p_hold_a + (1 - p_hold_b)) / 2
                p_markov_A = p_game_A**2 * (3 - 2 * p_game_A)
                
                # Peso: 40% Stats, 60% Elo (più affidabile)
                elo_diff = elo_a - elo_b
                p_elo_A = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
                p_final_A = (p_markov_A * 0.40) + (p_elo_A * 0.60)
                
                margine_book = (1/q_a) + (1/q_b)
                edge_book = p_final_A - ((1/q_a) / margine_book)
                margine_pin = (1/q_pin_a) + (1/q_pin_b)
                edge_pin = p_final_A - ((1/q_pin_a) / margine_pin)
                
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prob. Finale A", f"{p_final_A:.1%}", f"(Stats: {p_markov_A:.1%} | Elo: {p_elo_A:.1%})")
                c2.metric("Quota Fair", f"@{1/p_final_A:.2f}")
                c3.metric("Edge vs TUO BOOK", f"{edge_book*100:+.1f}%")
                c4.metric("Edge vs PINNACLE", f"{edge_pin*100:+.1f}%")
                
                if edge_book > 0.05 and abs(edge_pin) < 0.08:
                    st.success("🔥 VALORE REALE: Scommetti.")
                elif edge_book > 0.05 and abs(edge_pin) >= 0.08:
                    st.warning("⚠️ TRAPPOLA: Il book sembra generoso, ma Pinnacle/Elo dicono no.")
                else:
                    st.error("❌ NESSUN VALORE.")

# ================= TAB 2: GOOGLE SHEETS =================
with tab2:
    st.subheader("Analisi Massiva da Foglio Google")
    st.markdown("Incolla qui il link CSV del tuo foglio. *Nota: il foglio deve avere le colonne delle stats + 'Rank_A' e 'Rank_B'*")
    url_sheet = st.text_input("Link CSV Google Sheet", key="t2_url", placeholder="https://docs.google.com/.../pub?output=csv")
    
    if st.button("📊 Carica e Analizza Foglio", type="secondary", key="btn_2"):
        if url_sheet:
            try:
                df = pd.read_csv(url_sheet)
                required = ['A_first_in', 'A_first_w', 'A_second_w', 'A_ret_first', 'A_ret_second', 
                            'B_first_in', 'B_first_w', 'B_second_w', 'B_ret_first', 'B_ret_second', 
                            'Quota_A', 'Quota_B', 'Rank_A', 'Rank_B']
                if not all(col in df.columns for col in required):
                    st.error("Colonne mancanti nel foglio. Servono tutte le stats + Rank_A e Rank_B.")
                else:
                    results = []
                    for _, row in df.iterrows():
                        p_point_a = calc_p_point(row['A_first_in'], row['A_first_w'], row['A_second_w'], row['A_ret_first'], row['A_ret_second'])
                        p_hold_a = hold_probability(p_point_a)
                        p_point_b = calc_p_point(row['B_first_in'], row['B_first_w'], row['B_second_w'], row['B_ret_first'], row['B_ret_second'])
                        p_hold_b = hold_probability(p_point_b)
                        
                        p_game_A = (p_hold_a + (1 - p_hold_b)) / 2
                        p_markov_A = p_game_A**2 * (3 - 2 * p_game_A)
                        
                        # Stima Elo dal Rank per il foglio (Rank 1=2000, Rank 200=1500)
                        elo_a_est = max(1200, 2000 - (float(row['Rank_A']) * 2.5))
                        elo_b_est = max(1200, 2000 - (float(row['Rank_B']) * 2.5))
                        p_elo_A = 1.0 / (1.0 + 10.0 ** (-(elo_a_est - elo_b_est) / 400.0))
                        
                        p_final_A = (p_markov_A * 0.40) + (p_elo_A * 0.60)
                        
                        q_a, q_b = float(row['Quota_A']), float(row['Quota_B'])
                        margine = (1/q_a) + (1/q_b)
                        edge = p_final_A - ((1/q_a) / margine)
                        
                        results.append({
                            'Partita': f"Rank {int(row['Rank_A'])} vs {int(row['Rank_B'])}",
                            'Prob. A': f"{p_final_A:.1%}",
                            'Quota Fair': f"@{1/p_final_A:.2f}",
                            'Quota Book': f"@{q_a}",
                            'Edge %': f"{edge*100:+.1f}%",
                            'Verdetto': "✅ VALUE" if edge > 0.05 else "❌ NO"
                        })
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
            except Exception as e:
                st.error(f"Errore: {e}")
