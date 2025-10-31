"""
Streamlit app: Illiquid Option / Predatory Algo Simulation
Save this file as `streamlit_predatory_sim.py` and run with:
    pip install streamlit pandas matplotlib
    streamlit run streamlit_predatory_sim.py

Educational/demo only. Do NOT use for market manipulation.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Illiquid Option + Algo Simulation", layout="wide")

st.title("Illiquid Option Market — Algo vs Human Simulation (Educational)")
st.markdown(
    """
    This interactive demo simulates a simplified illiquid options market where an algorithmic participant
    can both buy and sell to create momentum and then flip to sell to a human liquidity taker.

    ⚠️ **Ethics & legal:** This notebook is for educational and research purposes only. Any attempt to
    manipulate real financial markets is illegal and unethical.
    """
)

# Sidebar controls
st.sidebar.header("Market / Simulation Parameters")
initial_bid = st.sidebar.number_input("Initial bid", value=20.0, step=1.0, format="%.2f")
initial_ask = st.sidebar.number_input("Initial ask", value=100.0, step=1.0, format="%.2f")
fair_price = st.sidebar.number_input("Fair price", value=40.0, step=1.0, format="%.2f")
threshold_pct = st.sidebar.slider("Algo flip threshold (% above fair)", 0, 200, 20) / 100.0
human_buy_intent = st.sidebar.number_input("Human intended buy price", value=21.0, format="%.2f")
max_steps = st.sidebar.slider("Max simulation steps", 10, 500, 60)

st.sidebar.markdown("---")
st.sidebar.header("Algo Behavior Adjustments")
# control how aggressively algo raises bid (0-1)
algo_aggression = st.sidebar.slider("Algo buy aggressiveness (0=gentle, 1=aggressive)", 0.0, 1.0, 0.35)
# how many units per algo trade
algo_trade_size = st.sidebar.number_input("Algo trade size (units per action)", value=1.0, step=1.0)
human_max_units = st.sidebar.number_input("Human max units to buy", value=3.0, step=1.0)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run simulation")

# Helper: simulation function

def run_simulation(initial_bid, initial_ask, fair_price, threshold_pct, human_buy_intent, max_steps,
                   algo_aggression, algo_trade_size, human_max_units):
    events = []
    bid = float(initial_bid)
    ask = float(initial_ask)
    threshold = fair_price * (1 + threshold_pct)
    last_price = None

    def record(step, actor, action, price, bid, ask, note=""):
        events.append({
            "step": step,
            "actor": actor,
            "action": action,
            "price": round(price, 2) if price is not None else None,
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "note": note,
        })

    record(0, "market", "initial_quotes", None, bid, ask, "illiquid wide spread")
    record(1, "human", "place_buy_intent", human_buy_intent, bid, ask, "human willing to buy")

    step = 2
    algo_side = "buy"
    human_position = 0.0
    algo_position = 0.0

    while step < max_steps:
        if algo_side == "buy":
            # algo posts a higher bid depending on aggression
            delta = max(0.2, (human_buy_intent - bid) * algo_aggression)
            new_bid = bid + delta
            trade_price = new_bid
            last_price = trade_price
            bid = new_bid
            # keep ask far unless algo chooses to tighten
            ask = max(ask, last_price + 15)
            algo_position += algo_trade_size
            record(step, "algo", "buy_to_push", trade_price, bid, ask, "algo buys to create momentum")
            step += 1

            # human may get a fill if price is within a tolerance
            if trade_price <= human_buy_intent + 30 and human_position < human_max_units:
                human_buy_price = max(human_buy_intent, trade_price)
                human_position += 1.0
                record(step, "human", "buy_filled", human_buy_price, bid, ask, "human receives partial fill")
                step += 1

            if last_price >= threshold:
                algo_side = "sell"
                record(step, "algo", "flip_to_sell", None, bid, ask, f"price reached threshold {threshold:.2f}")
                step += 1

        else:  # sell mode
            # algo posts a sell near the elevated last_price
            sell_price = max(ask - 3, (last_price or ask) * 1.02)
            last_price = sell_price
            ask = sell_price
            algo_position -= algo_trade_size
            record(step, "algo", "sell_to_human_or_book", sell_price, bid, ask, "algo sells into demand")
            step += 1

            # Human might still buy more at the worse price
            if human_position < human_max_units:
                human_position += 1.0
                record(step, "human", "buy_filled_at_loss", sell_price, bid, ask,
                       "human buys at elevated price (loss vs fair)")
                step += 1

            # after profitable extraction, algo resets quotes
            if algo_position <= 0:
                bid = float(initial_bid)
                ask = float(initial_ask)
                record(step, "algo", "reset_quotes", None, bid, ask, "algo returns to wide quotes")
                step += 1
                break

    record(step, "summary", "positions", None, bid, ask,
           f"human_position={human_position}, algo_position={algo_position}, last_price={last_price}")

    df = pd.DataFrame(events)
    return df


if run_button:
    df = run_simulation(initial_bid, initial_ask, fair_price, threshold_pct,
                        human_buy_intent, max_steps, algo_aggression, algo_trade_size, human_max_units)

    st.subheader("Simulation timeline")
    st.dataframe(df)

    # Plot trade prices over step index
    trade_rows = df[df['price'].notnull()]
    if not trade_rows.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(trade_rows['step'], trade_rows['price'], marker='o')
        ax.set_xlabel('Step')
        ax.set_ylabel('Trade Price')
        ax.set_title('Simulated Trade Prices Over Steps')
        ax.axhline(fair_price, linestyle='--', label='Fair price')
        ax.axhline(fair_price * (1 + threshold_pct), linestyle=':', label='Threshold')
        ax.legend()
        st.pyplot(fig)

    # Provide CSV download
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    st.download_button("Download events CSV", data=csv_bytes, file_name="simulation_events.csv", mime='text/csv')

    st.markdown("---")
    st.info("This is a synthetic model. Real markets include much more complexity (order books, multiple participants, latency, fees, regulation).")

else:
    st.info('Adjust parameters in the sidebar and click **Run simulation**.')

st.markdown("---")
st.caption("Created for educational demonstration — do not implement manipulative strategies in real markets.")
