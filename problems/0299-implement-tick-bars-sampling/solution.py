def tick_bars(ticks: list, bar_size: int) -> list:
    """
    Sample tick data into tick bars.

    Args:
        ticks: List of tuples (timestamp, price, volume) representing individual trades
        bar_size: Number of ticks per bar

    Returns:
        List of dictionaries with keys: 'timestamp', 'open', 'high', 'low', 'close', 'volume'
    """
    # Validate inputs
    if not ticks or not isinstance(bar_size, int) or bar_size <= 0:
        return []

    n_complete = len(ticks) // bar_size
    if n_complete == 0:
        return []

    bars = []
    for i in range(n_complete):
        chunk = ticks[i * bar_size : (i + 1) * bar_size]

        # unpack chunk
        ts_last = chunk[-1][0]
        prices = [float(t[1]) for t in chunk]
        vols = [float(t[2]) for t in chunk]

        o = prices[0]
        h = max(prices)
        l = min(prices)
        c = prices[-1]
        v = sum(vols)

        bars.append({
            "timestamp": ts_last,
            "open": round(o, 2),
 