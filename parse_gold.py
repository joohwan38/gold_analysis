
import re

def parse_gold_trades(file_path):
    """
    Parses the gold.txt file to extract trade data, calculates cost-effectiveness,
    and returns a sorted list of trades.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    trades = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for gold amount lines
        if 'gold' in line and line.replace(' ', '').replace('\t', '').endswith('gold'):
            gold_match = re.search(r'(\d+)\s+gold', line)
            if gold_match:
                quantity = int(gold_match.group(1))

                # Look for the next line with Duration, Seller, Price pattern
                j = i + 1
                while j < len(lines) and j < i + 3:  # Check next 2 lines
                    next_line = lines[j].strip()

                    # Pattern: Duration Seller Faction Price
                    trade_match = re.search(r'(Long|Medium|Short)\s+([A-Za-z0-9]+)\s+.*?(\d+)\s+coins', next_line)
                    if trade_match:
                        duration = trade_match.group(1)
                        seller = trade_match.group(2)
                        price = int(trade_match.group(3))

                        if price > 0:
                            gold_per_coin = round(quantity / price, 1)
                            trades.append({
                                "seller": seller,
                                "quantity": quantity,
                                "price": price,
                                "gpc": gold_per_coin,
                                "duration": duration
                            })
                        break
                    j += 1
        i += 1

    # Sort by gold_per_coin (gpc) in descending order
    sorted_trades = sorted(trades, key=lambda x: x['gpc'], reverse=True)
    return sorted_trades

def print_trades_table(trades):
    """Prints the list of trades in a formatted table."""
    print(f"{'판매자 (Seller)':<15} | {'총 골드 (Gold)':>12} | {'가격 (Coins)':>12} | {'코인당 골드':>15} | {'기간':>8}")
    print("-" * 80)
    for trade in trades:
        print(f"{trade['seller']:<15} | {trade['quantity']:>12,} | {trade['price']:>12,} | {trade['gpc']:>15.1f} | {trade['duration']:>8}")

if __name__ == "__main__":
    # NOTE: Make sure the gold.txt file is in the same directory as this script,
    # or provide the full path to the file.
    file_name = "gold.txt"
    parsed_trades = parse_gold_trades(file_name)
    print_trades_table(parsed_trades)
