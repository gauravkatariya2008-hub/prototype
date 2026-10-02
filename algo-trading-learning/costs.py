"""
Approximate charges for equity DELIVERY trades on NSE through Zerodha.

These rates change from time to time. Check Zerodha's brokerage calculator
(zerodha.com/brokerage-calculator) and update them here when they do.
Every profit number this project prints is net of these charges, because
ignoring them is the fastest way to believe a losing strategy is a winner.
"""

BROKERAGE = 0.0                  # Zerodha charges zero brokerage on delivery
STT = 0.001                      # 0.1% on both buy and sell
EXCHANGE_TXN = 0.0000297         # NSE transaction charge
SEBI_FEE = 10 / 1e7              # Rs 10 per crore
STAMP_DUTY_BUY = 0.00015         # 0.015% on buy side only
GST = 0.18                       # on brokerage + exchange + SEBI fees
DP_CHARGE_PER_SELL = 15.34       # flat depository charge each time you sell a stock
ETF_STT_SELL = 0.00001           # equity ETFs (like NIFTYBEES): 0.001% on sell, nothing on buy


def trade_cost(value, side, etf=False):
    """
    Approximate total charges (in Rs) for buying or selling `value` rupees of stock.
    etf=True uses the much lower securities transaction tax on equity ETFs.
    """
    exchange = value * EXCHANGE_TXN
    sebi = value * SEBI_FEE
    stt = (ETF_STT_SELL if side == "sell" else 0.0) if etf else STT
    cost = value * stt + exchange + sebi + (BROKERAGE + exchange + sebi) * GST
    if side == "buy":
        cost += value * STAMP_DUTY_BUY
    else:
        cost += DP_CHARGE_PER_SELL
    return cost


def round_trip_cost(buy_value, sell_value):
    """Charges for a complete buy-then-sell trade."""
    return trade_cost(buy_value, "buy") + trade_cost(sell_value, "sell")
