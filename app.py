import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
import os

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
        return pd.DataFrame()

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
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sorted_amounts, best_gpcs, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Gold Amount')
    ax.set_ylabel('Best Gold per Coin')
    ax.set_title('Best Value Trend by Gold Amount')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

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

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(sellers, avg_gpcs, color='skyblue', alpha=0.7)
    ax.set_xlabel('Seller')
    ax.set_ylabel('Average Gold per Coin')
    ax.set_title('Top 10 Sellers Average Value')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, avg_gpcs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{value:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    return fig

# Global variable to store current trades data
current_trades = []

def process_text_input(text_input):
    """Process the text input and update global trades data"""
    global current_trades
    current_trades = parse_gold_trades_from_text(text_input)

    if current_trades:
        return f"✅ 총 {len(current_trades)}개의 거래를 파싱했습니다."
    else:
        return "⚠️ 파싱된 거래 데이터가 없습니다. 텍스트 형식을 확인해주세요."

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

# Custom CSS for better UI
custom_css = """
#warning-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
    color: white;
    text-align: center;
}
#warning-text {
    font-size: 16px;
    font-weight: 500;
    margin: 10px 0;
}
.main-title {
    text-align: center;
    background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
"""

# Create Gradio interface
with gr.Blocks(title="WoW 골드 거래 분석기", css=custom_css, theme=gr.themes.Soft()) as app:
    gr.Markdown("# 💰 WoW 골드 거래 분석기", elem_classes="main-title")

    # Sleep mode warning banner
    with gr.Row(elem_id="warning-box"):
        gr.Markdown("""
        ### ⏰ 서비스 안내

        이 앱은 Hugging Face Spaces 무료 플랜으로 호스팅됩니다.

        **30분 동안 사용하지 않으면 슬립 모드로 전환됩니다.**

        처음 접속 시 또는 슬립 모드 후 재접속 시 **앱 로딩에 30초~1분 정도 소요**될 수 있습니다.

        잠시만 기다려주세요! 🙏
        """, elem_id="warning-text")

    with gr.Tabs():
        # Tab 1: Text Input
        with gr.Tab("📝 데이터 입력"):
            gr.Markdown("## 골드 거래 데이터 입력")
            gr.Markdown("""
            ### 사용 방법:
            1. WoW 게임 내 골드 거래소에서 거래 데이터를 복사합니다.
            2. 아래 텍스트 박스에 붙여넣습니다.
            3. '🔄 데이터 처리' 버튼을 클릭합니다.
            4. '📊 분석 결과' 탭에서 결과를 확인합니다.

            #### 예시 형식:
            ```
            1000 gold
            Long PlayerName Horde 50 coins

            2000 gold
            Medium PlayerName2 Alliance 80 coins
            ```
            """)

            text_input = gr.Textbox(
                lines=20,
                placeholder="여기에 골드 거래 데이터를 붙여넣으세요...\n\n예시:\n1000 gold\nLong PlayerName Horde 50 coins\n\n2000 gold\nMedium PlayerName2 Alliance 80 coins",
                label="거래 데이터 텍스트"
            )

            with gr.Row():
                save_btn = gr.Button("💾 파일로 저장", variant="secondary")
                process_btn = gr.Button("🔄 데이터 처리", variant="primary", scale=2)

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
            gr.Markdown("데이터를 처리한 후 새로고침 버튼을 클릭하여 결과를 확인하세요.")

            with gr.Tabs():
                # Sub-tab 1: Trade Rankings
                with gr.Tab("🏆 가성비 순위"):
                    gr.Markdown("### 코인당 골드 기준 내림차순 정렬")
                    gr.Markdown("**GPC (Gold Per Coin)**: 1 코인당 얻을 수 있는 골드 양. 높을수록 가성비가 좋습니다.")

                    refresh_table_btn = gr.Button("🔄 테이블 새로고침", variant="primary")
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
                    gr.Markdown("골드 수량에 따른 최고 GPC(코인당 골드) 값을 차트로 확인하세요.")

                    with gr.Row():
                        refresh_trend_btn = gr.Button("🔄 가격 추이 차트 새로고침", variant="primary")
                        refresh_seller_btn = gr.Button("🔄 판매자 분석 새로고침", variant="primary")

                    with gr.Row():
                        with gr.Column():
                            trend_chart = gr.Plot(label="가격 추이 차트")
                        with gr.Column():
                            seller_chart = gr.Plot(label="판매자 분석 차트 (Top 10)")

                    refresh_trend_btn.click(
                        fn=get_price_trend,
                        outputs=[trend_chart]
                    )

                    refresh_seller_btn.click(
                        fn=get_seller_analysis,
                        outputs=[seller_chart]
                    )

        # Tab 3: About
        with gr.Tab("ℹ️ 정보"):
            gr.Markdown("""
            ## WoW 골드 거래 분석기

            이 도구는 World of Warcraft의 골드 거래 데이터를 분석하여 최고의 거래를 찾아줍니다.

            ### 주요 기능:
            - 📊 **거래 데이터 파싱**: 텍스트 형식의 거래 데이터를 자동으로 분석
            - 🏆 **가성비 순위**: 코인당 골드(GPC) 기준으로 거래를 정렬
            - 📈 **가격 추이 분석**: 골드 수량별 최고 가성비 시각화
            - 👥 **판매자 분석**: 평균 GPC 기준 상위 10명의 판매자 분석
            - 💾 **데이터 저장**: 분석한 데이터를 파일로 저장

            ### 기술 스택:
            - **Python**: 데이터 처리 및 분석
            - **Gradio**: 웹 인터페이스
            - **Pandas**: 데이터 처리
            - **Matplotlib**: 차트 시각화

            ### 호스팅:
            - **Hugging Face Spaces** - 무료 플랜
            - ⚠️ 30분 비활성 시 슬립 모드 (재시작 시 30초~1분 소요)

            ---

            Made with ❤️ for WoW players
            """)

if __name__ == "__main__":
    # Hugging Face Spaces용 설정
    app.launch()
