import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
import os
import matplotlib.font_manager as fm

# Set Korean font for matplotlib
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def parse_gold_trades_from_text(text_input):
    """
    Parses the input text to extract trade data, calculates cost-effectiveness,
    and returns a sorted list of trades.
    """
    if not text_input.strip():
        return []

    lines = text_input.strip().split('\n')
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

def save_text_to_file(text_input):
    """Save the input text to a timestamped file"""
    if not text_input.strip():
        return "텍스트가 비어있습니다."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gold_data_{timestamp}.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text_input)
        return f"파일이 저장되었습니다: {filename}"
    except Exception as e:
        return f"저장 실패: {str(e)}"

def create_trades_table(trades):
    """Create a formatted table of trades"""
    if not trades:
        return "데이터가 없습니다."

    # Create DataFrame for better display
    df = pd.DataFrame(trades)
    df = df[['seller', 'quantity', 'price', 'gpc', 'duration']]
    df.columns = ['판매자', '총 골드', '가격(코인)', '코인당 골드', '기간']

    # Format numbers with commas
    df['총 골드'] = df['총 골드'].apply(lambda x: f"{x:,}")
    df['가격(코인)'] = df['가격(코인)'].apply(lambda x: f"{x:,}")

    return df

def create_price_trend_chart(trades):
    """Create a price trend chart showing best prices over time"""
    if not trades:
        return None

    # Group by unique gold amounts and find the best price for each
    price_data = {}
    for trade in trades:
        gold_amount = trade['quantity']
        gpc = trade['gpc']

        if gold_amount not in price_data or gpc > price_data[gold_amount]['best_gpc']:
            price_data[gold_amount] = {
                'best_gpc': gpc,
                'seller': trade['seller'],
                'price': trade['price']
            }

    # Sort by gold amount
    sorted_amounts = sorted(price_data.keys())
    best_gpcs = [price_data[amount]['best_gpc'] for amount in sorted_amounts]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_amounts, best_gpcs, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Gold Amount')
    plt.ylabel('Best Gold per Coin')
    plt.title('Best Value Trend by Gold Amount')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt

def analyze_sellers(trades):
    """Analyze seller performance"""
    if not trades:
        return None

    # Group by seller and calculate average GPC
    seller_stats = {}
    for trade in trades:
        seller = trade['seller']
        if seller not in seller_stats:
            seller_stats[seller] = {'total_gpc': 0, 'count': 0, 'total_gold': 0}

        seller_stats[seller]['total_gpc'] += trade['gpc']
        seller_stats[seller]['count'] += 1
        seller_stats[seller]['total_gold'] += trade['quantity']

    # Calculate averages and sort
    seller_analysis = []
    for seller, stats in seller_stats.items():
        avg_gpc = stats['total_gpc'] / stats['count']
        seller_analysis.append({
            'seller': seller,
            'avg_gpc': round(avg_gpc, 1),
            'trade_count': stats['count'],
            'total_gold': stats['total_gold']
        })

    seller_analysis.sort(key=lambda x: x['avg_gpc'], reverse=True)

    # Create top 10 sellers chart
    top_sellers = seller_analysis[:10]
    sellers = [s['seller'] for s in top_sellers]
    avg_gpcs = [s['avg_gpc'] for s in top_sellers]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(sellers, avg_gpcs, color='skyblue', alpha=0.7)
    plt.xlabel('Seller')
    plt.ylabel('Average Gold per Coin')
    plt.title('Top 10 Sellers Average Value')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, avg_gpcs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{value:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    return plt

# Global variable to store current trades data
current_trades = []

def process_text_input(text_input):
    """Process the text input and update global trades data"""
    global current_trades
    current_trades = parse_gold_trades_from_text(text_input)

    if current_trades:
        return f"총 {len(current_trades)}개의 거래를 파싱했습니다."
    else:
        return "파싱된 거래 데이터가 없습니다. 텍스트 형식을 확인해주세요."

def get_trades_table():
    """Get the current trades table"""
    global current_trades
    return create_trades_table(current_trades)

def get_price_trend():
    """Get the price trend chart"""
    global current_trades
    return create_price_trend_chart(current_trades)

def get_seller_analysis():
    """Get the seller analysis chart"""
    global current_trades
    return analyze_sellers(current_trades)

# Create Gradio interface
with gr.Blocks(title="WoW 골드 거래 분석기") as app:
    gr.Markdown("# WoW 골드 거래 분석기")

    with gr.Tabs():
        # Tab 1: Text Input
        with gr.Tab("📝 데이터 입력"):
            gr.Markdown("## 골드 거래 데이터 입력")
            gr.Markdown("gold.txt와 같은 형식의 텍스트를 붙여넣으세요.")

            text_input = gr.Textbox(
                lines=20,
                placeholder="여기에 골드 거래 데이터를 붙여넣으세요...",
                label="거래 데이터 텍스트"
            )

            with gr.Row():
                save_btn = gr.Button("💾 파일로 저장", variant="secondary")
                process_btn = gr.Button("🔄 데이터 처리", variant="primary")

            status_output = gr.Textbox(label="상태", interactive=False)

            # Event handlers for Tab 1
            save_btn.click(
                fn=save_text_to_file,
                inputs=[text_input],
                outputs=[status_output]
            )

            process_btn.click(
                fn=process_text_input,
                inputs=[text_input],
                outputs=[status_output]
            )

        # Tab 2: Results
        with gr.Tab("📊 분석 결과"):
            gr.Markdown("## 거래 분석 결과")

            with gr.Tabs():
                # Sub-tab 1: Trade Rankings
                with gr.Tab("🏆 가성비 순위"):
                    gr.Markdown("### 코인당 골드 기준 내림차순 정렬")

                    refresh_table_btn = gr.Button("🔄 테이블 새로고침")
                    trades_table = gr.Dataframe(
                        headers=["판매자", "총 골드", "가격(코인)", "코인당 골드", "기간"],
                        label="거래 순위"
                    )

                    refresh_table_btn.click(
                        fn=get_trades_table,
                        outputs=[trades_table]
                    )

                # Sub-tab 2: Price Trends
                with gr.Tab("📈 가격 추이"):
                    gr.Markdown("### 골드 양별 최고 가성비 추이")

                    with gr.Row():
                        refresh_trend_btn = gr.Button("🔄 차트 새로고침")
                        refresh_seller_btn = gr.Button("🔄 판매자 분석 새로고침")

                    with gr.Row():
                        trend_chart = gr.Plot(label="가격 추이 차트")
                        seller_chart = gr.Plot(label="판매자 분석 차트")

                    refresh_trend_btn.click(
                        fn=get_price_trend,
                        outputs=[trend_chart]
                    )

                    refresh_seller_btn.click(
                        fn=get_seller_analysis,
                        outputs=[seller_chart]
                    )

if __name__ == "__main__":
    app.launch(share=False, inbrowser=True)