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
st.title("🎾 Tennis Quant Model: Edge Finder")
st.markdown("Modello probabilistico Markoviano per l'identificazione di valore nelle quote.")

st.divider()

# SEZIONE INPUT MANUALI
st.subheader("⚙️ Modalità Match: Giocatore A vs Giocatore B")
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🟦 Giocatore A (Al Servizio)")
    a_first_in = st.number_input("Prima in campo (0-1)", 0.0, 1.0, 0.62, step=0.01, key="a_first_in")
    a_first_w = st.number_input("Punti vinti con la prima", 0.0, 1.0, 0.75, step=0.01, key="a_first_w")
    a_second_w = st.number_input("Punti vinti con la seconda", 0.0, 1.0, 0.50, step=0.01, key="a_second_w")
    a_ret_first = st.number_input("Punti risposta A alla prima di B", 0.0, 1.0, 0.32, step=0.01, key="a_ret_first")
    a_ret_second = st.number_input("Punti risposta A alla seconda di B", 0.0, 1.0, 0.50, step=0.01, key="a_ret_second")

with col2:
    st.markdown("##### 🟥 Giocatore B (Al Servizio)")
    b_first_in = st.number_input("Prima in campo (0-1)", 0.0, 1.0, 0.60, step=0.01, key="b_first_in")
    b_first_w = st.number_input("Punti vinti con la prima", 0.0, 1.0, 0.72, step=0.01, key="b_first_w")
    b_second_w = st.number_input("Punti vinti con la seconda", 0.0, 1.0, 0.48, step=0.01, key="b_second_w")
    b_ret_first = st.number_input("Punti risposta B alla prima di A", 0.0, 1.0, 0.30, step=0.01, key="b_ret_first")
    b_ret_second = st.number_input("Punti risposta B alla seconda di A", 0.0, 1.0, 0.52, step=0.01, key="b_ret_second")

# CALCOLI
p_point_a = calc_p_point(a_first_in, a_first_w, a_second_w, b_ret_first, b_ret_second)
p_hold_a = hold_probability(p_point_a)

p_point_b = calc_p_point(b_first_in, b_first_w, b_second_w, a_ret_first, a_ret_second)
p_hold_b = hold_probability(p_point_b)

st.divider()

# SEZIONE RISULTATI
st.subheader("📊 Risultati del Modello")
res_col1, res_col2, res_col3 = st.columns(3)

with res_col1:
    st.metric("P(Point) Giocatore A", f"{p_point_a:.1%}")
    st.metric("P(Hold) Giocatore A", f"{p_hold_a:.1%}")
    st.metric("Quota Fair Hold A", f"@{1/p_hold_a:.3f}")

with res_col2:
    st.metric("P(Point) Giocatore B", f"{p_point_b:.1%}")
    st.metric("P(Hold) Giocatore B", f"{p_hold_b:.1%}")
    st.metric("Quota Fair Hold B", f"@{1/p_hold_b:.3f}")

with res_col3:
    p_set_a = p_hold_a**2 * (3 - 2*p_hold_a) # Approssimazione Best of 3 semplificata
    p_set_b = 1 - p_set_a
    st.metric("P(Vittoria Match) A", f"{p_set_a:.1%}")
    st.metric("P(Vittoria Match) B", f"{p_set_b:.1%}")
    st.metric("Quota Fair Match A", f"@{1/p_set_a:.3f}")

st.divider()

# SEZIONE EDGE CALCULATOR
st.subheader("💰 Calcolatore Edge vs Bookmaker")
edge_col1, edge_col2 = st.columns(2)

with edge_col1:
    st.markdown("##### Giocatore A (Match Winner)")
    q_book_a = st.number_input("Quota Bookmaker A", 1.01, 10.0, 1.65, step=0.01, key="q_book_a")
    prob_impl_a = 1 / q_book_a
    # No-vig semplificato su 2 vie
    q_book_b_temp = st.number_input("Quota Bookmaker B (per no-vig)", 1.01, 10.0, 2.35, step=0.01, key="q_book_b_temp")
    margine = (1/q_book_a) + (1/q_book_b_temp)
    prob_fair_a = prob_impl_a / margine
    edge_a = p_set_a - prob_fair_a
    ev_a = (p_set_a * (q_book_a - 1)) - (1 - p_set_a)
    
    st.write(f"Prob. Fair (No-Vig): **{prob_fair_a:.1%}**")
    st.write(f"Edge: **{edge_a:.1%}**")
    if edge_a > 0:
        st.success(f"✅ VALUE BET (EV: {ev_a:.1%})")
    else:
        st.error("❌ NO VALUE")

with edge_col2:
    st.markdown("##### Total Games Over/Under")
    line_games = st.number_input("Linea Games", 10.5, 30.5, 21.5, step=1.0, key="line_games")
    q_book_over = st.number_input("Quota Bookmaker Over", 1.01, 10.0, 1.50, step=0.01, key="q_book_over")
    q_book_under = st.number_input("Quota Bookmaker Under", 1.01, 10.0, 2.60, step=0.01, key="q_book_under")
    
    # Simulazione semplificata prob over (in un modello reale verrebbe dalla convoluzione delle distribuzioni)
    # Qui usiamo un'euristica basata sul differenziale di hold per demo
    prob_model_over = 0.65 if (p_hold_a + p_hold_b) > 1.55 else 0.45 
    
    margine_games = (1/q_book_over) + (1/q_book_under)
    prob_fair_over = (1/q_book_over) / margine_games
    edge_over = prob_model_over - prob_fair_over
    ev_over = (prob_model_over * (q_book_over - 1)) - (1 - prob_model_over)
    
    st.write(f"Prob. Fair Over (No-Vig): **{prob_fair_over:.1%}**")
    st.write(f"Prob. Modello Over (Stima): **{prob_model_over:.1%}**")
    st.write(f"Edge: **{edge_over:.1%}**")
    if edge_over > 0:
        st.success(f"✅ VALUE BET (EV: {ev_over:.1%})")
    else:
        st.error("❌ NO VALUE")