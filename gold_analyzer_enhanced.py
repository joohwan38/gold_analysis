import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import re
import sqlite3
import os
import glob
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Tuple, Optional

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GoldTradeDatabase:
    """SQLite database manager for gold trade data"""

    def __init__(self, db_path: str = "gold_trades.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with required tables and ensure schema updates"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                seller TEXT,
                quantity INTEGER,
                price INTEGER,
                gold_per_coin REAL,
                duration TEXT,
                source_file TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Historical snapshots table for tracking market state over time
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                best_gpc REAL,
                avg_gpc REAL,
                median_gpc REAL,
                total_trades INTEGER,
                total_volume INTEGER,
                unique_sellers INTEGER,
                source_file TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Price trends table for efficient querying
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                quantity_range TEXT,
                min_gpc REAL,
                max_gpc REAL,
                avg_gpc REAL,
                trade_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Add best_gpc_filtered column if it doesn't exist
        cursor.execute("PRAGMA table_info(market_snapshots)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'best_gpc_filtered' not in columns:
            cursor.execute("ALTER TABLE market_snapshots ADD COLUMN best_gpc_filtered REAL")

        # Add sequence column if it doesn't exist
        if 'sequence' not in columns:
            cursor.execute("ALTER TABLE market_snapshots ADD COLUMN sequence INTEGER")

        # Add quantity range columns if they don't exist
        range_columns = ['best_gpc_under_10k', 'best_gpc_under_20k', 'best_gpc_under_30k',
                        'best_gpc_under_50k', 'best_gpc_over_50k']
        for col in range_columns:
            if col not in columns:
                cursor.execute(f"ALTER TABLE market_snapshots ADD COLUMN {col} REAL")

        conn.commit()
        conn.close()

    def insert_trades(self, trades: List[Dict], timestamp: datetime, source_file: str):
        """Insert trades into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for trade in trades:
            cursor.execute('''
                INSERT INTO trades (timestamp, seller, quantity, price, gold_per_coin, duration, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                trade['seller'],
                trade['quantity'],
                trade['price'],
                trade['gpc'],
                trade['duration'],
                source_file
            ))

        conn.commit()
        conn.close()

    def insert_market_snapshot(self, snapshot: Dict, timestamp: datetime, source_file: str):
        """Insert market snapshot for time series analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO market_snapshots (
                timestamp, best_gpc, avg_gpc, median_gpc,
                total_trades, total_volume, unique_sellers, source_file, best_gpc_filtered, sequence,
                best_gpc_under_10k, best_gpc_under_20k, best_gpc_under_30k,
                best_gpc_under_50k, best_gpc_over_50k
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            snapshot['best_gpc'],
            snapshot['avg_gpc'],
            snapshot['median_gpc'],
            snapshot['total_trades'],
            snapshot['total_volume'],
            snapshot['unique_sellers'],
            source_file,
            snapshot['best_gpc_filtered'],
            snapshot.get('sequence', None),
            snapshot.get('best_gpc_under_10k', None),
            snapshot.get('best_gpc_under_20k', None),
            snapshot.get('best_gpc_under_30k', None),
            snapshot.get('best_gpc_under_50k', None),
            snapshot.get('best_gpc_over_50k', None)
        ))

        conn.commit()
        conn.close()

    def get_time_series_data(self, limit: int = None) -> pd.DataFrame:
        """Get time series data for market trends ordered by sequence"""
        conn = sqlite3.connect(self.db_path)

        if limit:
            query = '''
                SELECT * FROM market_snapshots
                WHERE sequence IS NOT NULL
                ORDER BY sequence
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(limit,))
        else:
            query = '''
                SELECT * FROM market_snapshots
                WHERE sequence IS NOT NULL
                ORDER BY sequence
            '''
            df = pd.read_sql_query(query, conn)

        conn.close()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

        return df

    def get_seller_performance_history(self, seller: str = None) -> pd.DataFrame:
        """Get historical performance data for sellers"""
        conn = sqlite3.connect(self.db_path)

        if seller:
            query = '''
                SELECT timestamp, seller, AVG(gold_per_coin) as avg_gpc,
                       COUNT(*) as trade_count, SUM(quantity) as total_volume
                FROM trades
                WHERE seller = ?
                GROUP BY timestamp, seller
                ORDER BY timestamp
            '''
            df = pd.read_sql_query(query, conn, params=(seller,))
        else:
            query = '''
                SELECT timestamp, seller, AVG(gold_per_coin) as avg_gpc,
                       COUNT(*) as trade_count, SUM(quantity) as total_volume
                FROM trades
                GROUP BY timestamp, seller
                ORDER BY timestamp, avg_gpc DESC
            '''
            df = pd.read_sql_query(query, conn)

        conn.close()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

        return df

class GoldTradeAnalyzer:
    """Enhanced gold trade analyzer with time series capabilities"""

    def __init__(self):
        self.db = GoldTradeDatabase()
        self.current_trades = []
        self.current_timestamp = None

    def parse_gold_trades_from_text(self, text_input: str) -> List[Dict]:
        """Parse trade data from text input"""
        if not text_input.strip():
            return []

        lines = text_input.strip().split('\n')
        trades = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Look for gold amount lines - more flexible pattern
            gold_match = re.search(r'(\d+)\s+gold', line)
            if gold_match:
                quantity = int(gold_match.group(1))

                # Look for the next line with Duration, Seller, Price pattern
                j = i + 1
                while j < len(lines) and j < i + 5:  # Check up to 4 lines ahead
                    next_line = lines[j].strip()

                    # Skip empty lines
                    if not next_line:
                        j += 1
                        continue

                    # More flexible pattern for Duration Seller [Faction] Price
                    # Handle cases with tabs, multiple spaces, and empty faction fields

                    # First try: strict pattern with tabs
                    trade_match = re.search(r'(Long|Medium|Short)\s*\t+\s*([A-Za-z0-9]+)\s*\t*\s*\t+\s*(\d+)\s+coins', next_line)

                    if not trade_match:
                        # Second try: flexible pattern with spaces
                        trade_match = re.search(r'(Long|Medium|Short)\s+([A-Za-z0-9]+)\s+.*?(\d+)\s+coins', next_line)

                    if not trade_match:
                        # Third try: very flexible pattern
                        trade_match = re.search(r'(Long|Medium|Short)[\s\t]+([A-Za-z0-9]+)[\s\t]+.*?(\d+)[\s\t]+coins', next_line)

                    if trade_match:
                        duration = trade_match.group(1)
                        seller = trade_match.group(2)
                        price = int(trade_match.group(3))

                        if price > 0 and quantity > 0:
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

    def extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract timestamp from filename"""
        # Pattern: gold_data_YYYYMMDD_HHMMSS.txt
        match = re.search(r'gold_data_(\d{8})_(\d{6})\.txt', filename)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        return None

    def calculate_market_snapshot(self, trades: List[Dict]) -> Dict:
        """Calculate market snapshot metrics"""
        if not trades:
            return {}

        gpcs = [trade['gpc'] for trade in trades]
        sellers = set(trade['seller'] for trade in trades)
        total_volume = sum(trade['quantity'] for trade in trades)

        # Filter trades for best price tracking (>= 10,000 gold)
        filtered_trades = [trade for trade in trades if trade['quantity'] >= 10000]
        best_gpc_filtered = max([trade['gpc'] for trade in filtered_trades]) if filtered_trades else 0.0

        # Calculate best GPC by quantity ranges
        ranges = {
            'under_10k': [t for t in trades if t['quantity'] < 10000],
            'under_20k': [t for t in trades if 10000 <= t['quantity'] < 20000],
            'under_30k': [t for t in trades if 20000 <= t['quantity'] < 30000],
            'under_50k': [t for t in trades if 30000 <= t['quantity'] < 50000],
            'over_50k': [t for t in trades if t['quantity'] >= 50000]
        }

        gpc_by_range = {}
        for range_name, range_trades in ranges.items():
            if range_trades:
                gpc_by_range[f'best_gpc_{range_name}'] = max([t['gpc'] for t in range_trades])
            else:
                gpc_by_range[f'best_gpc_{range_name}'] = None

        return {
            'best_gpc': max(gpcs),
            'avg_gpc': round(np.mean(gpcs), 2),
            'median_gpc': round(np.median(gpcs), 2),
            'total_trades': len(trades),
            'total_volume': total_volume,
            'unique_sellers': len(sellers),
            'best_gpc_filtered': best_gpc_filtered,
            **gpc_by_range  # Add range-based GPC metrics
        }

    def import_txt_files(self, force_refresh: bool = False) -> str:
        """Import all txt files in the directory, keeping only the latest per day"""
        txt_files = glob.glob("gold*.txt")

        # Group files by date and find the latest for each date
        files_by_date = {} # Key: YYYYMMDD, Value: (latest_timestamp, filename)
        for file_path in txt_files:
            timestamp = self.extract_timestamp_from_filename(file_path)
            if not timestamp:
                # For files without timestamp, use file modification time
                timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))

            date_key = timestamp.strftime("%Y%m%d")

            if date_key not in files_by_date or timestamp > files_by_date[date_key][0]:
                files_by_date[date_key] = (timestamp, file_path)

        # Sort by timestamp and get only the latest files for each day
        sorted_items = sorted(files_by_date.values(), key=lambda x: x[0])
        files_to_process = [item[1] for item in sorted_items]

        # If force_refresh, clear existing data first
        if force_refresh:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM market_snapshots")
            conn.commit()
            conn.close()

        imported_count = 0
        sequence = 0

        for file_path in files_to_process:
            timestamp = self.extract_timestamp_from_filename(file_path)
            if not timestamp:
                timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                trades = self.parse_gold_trades_from_text(content)
                if trades:
                    # Check if this file was already imported (unless force_refresh)
                    if not force_refresh:
                        conn = sqlite3.connect(self.db.db_path)
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM trades WHERE source_file = ?", (file_path,))
                        exists = cursor.fetchone()[0] > 0
                        conn.close()

                        if exists:
                            continue

                    self.db.insert_trades(trades, timestamp, file_path)

                    # Calculate and store market snapshot with sequence number
                    snapshot = self.calculate_market_snapshot(trades)
                    if snapshot:
                        snapshot['sequence'] = sequence
                        self.db.insert_market_snapshot(snapshot, timestamp, file_path)
                        sequence += 1

                    imported_count += 1

            except Exception as e:
                print(f"Error importing {file_path}: {e}")

        return f"Successfully imported {imported_count} files to database"

    def create_time_series_chart(self, limit: int = None) -> go.Figure:
        """Create time series chart showing market trends by sequence"""
        df = self.db.get_time_series_data(limit)

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No time series data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Create categorical date labels (YYMMDD format)
        df['date_label'] = df['timestamp'].dt.strftime('%y%m%d')
        df['full_date'] = df['timestamp'].dt.strftime('%Y-%m-%d')

        # Create subplots - now 4 rows instead of 3
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Gold per Coin Trends', 'GPC by Quantity Range', 'Trading Volume', 'Market Activity'),
            vertical_spacing=0.06
        )

        # Row 1: Best GPC trend with labels
        fig.add_trace(
            go.Scatter(x=df['date_label'], y=df['best_gpc_filtered'],
                      name='Best GPC (>=10k Gold)',
                      line=dict(color='green', width=3),
                      mode='lines+markers+text',
                      marker=dict(size=8),
                      text=df['best_gpc_filtered'].apply(lambda x: f'{x:.1f}' if pd.notna(x) else ''),
                      textposition='top center',
                      textfont=dict(size=9, color='green'),
                      hovertemplate='Date: %{customdata}<br>GPC: %{y:.1f}<extra></extra>',
                      customdata=df['full_date'],
                      connectgaps=False),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df['date_label'], y=df['avg_gpc'],
                      name='Avg GPC',
                      line=dict(color='blue', width=2),
                      mode='lines+markers+text',
                      marker=dict(size=6),
                      text=df['avg_gpc'].apply(lambda x: f'{x:.1f}'),
                      textposition='bottom center',
                      textfont=dict(size=8, color='blue'),
                      hovertemplate='Date: %{customdata}<br>GPC: %{y:.1f}<extra></extra>',
                      customdata=df['full_date'],
                      connectgaps=False),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df['date_label'], y=df['median_gpc'],
                      name='Median GPC',
                      line=dict(color='orange', width=2),
                      mode='lines+markers',
                      marker=dict(size=6),
                      hovertemplate='Date: %{customdata}<br>GPC: %{y:.1f}<extra></extra>',
                      customdata=df['full_date'],
                      connectgaps=False),
            row=1, col=1
        )

        # Row 2: GPC by Quantity Range (NEW)
        range_config = [
            ('best_gpc_under_10k', '1만골드 이하', '#FF6B6B'),
            ('best_gpc_under_20k', '2만골드 이하', '#4ECDC4'),
            ('best_gpc_under_30k', '3만골드 이하', '#45B7D1'),
            ('best_gpc_under_50k', '5만골드 이하', '#96CEB4'),
            ('best_gpc_over_50k', '5만골드 초과', '#FFEAA7')
        ]

        for col_name, label, color in range_config:
            if col_name in df.columns:
                # Filter out None values
                valid_data = df[df[col_name].notna()]
                if not valid_data.empty:
                    fig.add_trace(
                        go.Scatter(x=valid_data['date_label'], y=valid_data[col_name],
                                  name=label,
                                  line=dict(color=color, width=2),
                                  mode='lines+markers',
                                  hovertemplate=f'{label}<br>Date: ' + '%{customdata}<br>GPC: %{y:.1f}<extra></extra>',
                                  customdata=valid_data['full_date'],
                                  connectgaps=False),
                        row=2, col=1
                    )

        # Row 3: Total volume with labels
        fig.add_trace(
            go.Bar(x=df['date_label'], y=df['total_volume'],
                   name='Total Volume',
                   marker_color='lightblue',
                   hovertemplate='Date: %{customdata}<br>Volume: %{y:,}<extra></extra>',
                   customdata=df['full_date'],
                   text=df['total_volume'].apply(lambda x: f'{x/1000:.0f}k' if x >= 1000 else str(x)),
                   textposition='outside',
                   showlegend=False),
            row=3, col=1
        )

        # Row 4: Number of trades and sellers with labels
        fig.add_trace(
            go.Scatter(x=df['date_label'], y=df['total_trades'],
                      name='Total Trades',
                      mode='lines+markers+text',
                      line=dict(color='purple'),
                      marker=dict(size=8),
                      text=df['total_trades'],
                      textposition='top center',
                      textfont=dict(size=10),
                      hovertemplate='Date: %{customdata}<br>Trades: %{y}<extra></extra>',
                      customdata=df['full_date'],
                      connectgaps=False),
            row=4, col=1
        )

        fig.add_trace(
            go.Scatter(x=df['date_label'], y=df['unique_sellers'],
                      name='Unique Sellers',
                      mode='lines+markers+text',
                      line=dict(color='brown'),
                      marker=dict(size=8),
                      text=df['unique_sellers'],
                      textposition='bottom center',
                      textfont=dict(size=10),
                      hovertemplate='Date: %{customdata}<br>Sellers: %{y}<extra></extra>',
                      customdata=df['full_date'],
                      connectgaps=False),
            row=4, col=1
        )

        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Gold Market Time Series Analysis",
            showlegend=True
        )

        # Update x-axes to be categorical
        fig.update_xaxes(title_text="Date (YYMMDD)", row=4, col=1, type='category')
        fig.update_xaxes(type='category', row=1, col=1)
        fig.update_xaxes(type='category', row=2, col=1)
        fig.update_xaxes(type='category', row=3, col=1)

        fig.update_yaxes(title_text="Gold per Coin", row=1, col=1)
        fig.update_yaxes(title_text="Gold per Coin", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=4, col=1)

        return fig

    def create_price_volatility_chart(self) -> go.Figure:
        """Create price volatility analysis chart"""
        df = self.db.get_time_series_data(limit=30)  # Last 30 snapshots

        if df.empty or len(df) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for volatility analysis",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Calculate volatility metrics
        df['best_gpc_pct_change'] = df['best_gpc_filtered'].pct_change() * 100
        df['avg_gpc_pct_change'] = df['avg_gpc'].pct_change() * 100

        # Rolling volatility (standard deviation of returns)
        window = min(10, len(df) // 2)
        df['volatility'] = df['best_gpc_pct_change'].rolling(window=window).std()

        # Create labels
        df['date_label'] = df['timestamp'].dt.strftime('%y%m%d')
        df['full_date'] = df['timestamp'].dt.strftime('%Y-%m-%d')

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Changes (%)', 'Price Volatility'),
            vertical_spacing=0.1
        )

        # Price changes
        fig.add_trace(
            go.Scatter(x=df['date_label'], y=df['best_gpc_pct_change'],
                      name='Best GPC Change %',
                      mode='lines+markers',
                      hovertemplate='Date: %{customdata}<br>Change: %{y:.2f}%<extra></extra>',
                      customdata=df['full_date']),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df['date_label'], y=df['avg_gpc_pct_change'],
                      name='Avg GPC Change %',
                      mode='lines+markers',
                      hovertemplate='Date: %{customdata}<br>Change: %{y:.2f}%<extra></extra>',
                      customdata=df['full_date']),
            row=1, col=1
        )

        # Volatility
        fig.add_trace(
            go.Scatter(x=df['date_label'], y=df['volatility'],
                      name='Volatility',
                      fill='tonexty',
                      mode='lines',
                      hovertemplate='Date: %{customdata}<br>Volatility: %{y:.2f}<extra></extra>',
                      customdata=df['full_date']),
            row=2, col=1
        )

        fig.update_layout(
            height=600,
            title_text="Price Volatility Analysis"
        )

        fig.update_xaxes(title_text="Date (YYMMDD)", row=2, col=1, type='category')
        fig.update_xaxes(type='category', row=1, col=1)

        return fig

    def create_seller_performance_chart(self) -> go.Figure:
        """Create seller performance over time chart"""
        df = self.db.get_seller_performance_history()

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No seller performance data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Get top performers
        top_sellers = df.groupby('seller')['avg_gpc'].mean().nlargest(10).index

        fig = go.Figure()

        colors = px.colors.qualitative.Set3

        for i, seller in enumerate(top_sellers):
            seller_data = df[df['seller'] == seller].sort_values('timestamp')

            fig.add_trace(
                go.Scatter(
                    x=seller_data['timestamp'],
                    y=seller_data['avg_gpc'],
                    name=seller,
                    mode='lines+markers',
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{seller}</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Avg GPC: %{y}<br>' +
                                 'Trades: %{customdata[0]}<br>' +
                                 'Volume: %{customdata[1]:,}<extra></extra>',
                    customdata=seller_data[['trade_count', 'total_volume']].values
                )
            )

        fig.update_layout(
            title="Top Sellers Performance Over Time",
            xaxis_title="Time",
            yaxis_title="Average Gold per Coin",
            height=500,
            hovermode='closest'
        )

        return fig

    def create_gpc_density_chart(self) -> go.Figure:
        """Create GPC density plot for the latest date by quantity range"""
        # Get the latest snapshot
        df = self.db.get_time_series_data()

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available for density plot",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Get the most recent data point
        latest_row = df.iloc[-1]
        latest_date = latest_row['timestamp'].strftime('%Y-%m-%d')

        # Get all trades from the latest snapshot
        conn = sqlite3.connect(self.db.db_path)
        query = '''
            SELECT quantity, gold_per_coin
            FROM trades
            WHERE source_file = ?
            ORDER BY gold_per_coin DESC
        '''
        trades_df = pd.read_sql_query(query, conn, params=(latest_row['source_file'],))
        conn.close()

        if trades_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available for density plot",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Categorize trades by quantity range
        def categorize_quantity(qty):
            if qty < 10000:
                return '1만골드 이하'
            elif qty < 20000:
                return '2만골드 이하'
            elif qty < 30000:
                return '3만골드 이하'
            elif qty < 50000:
                return '5만골드 이하'
            else:
                return '5만골드 초과'

        trades_df['category'] = trades_df['quantity'].apply(categorize_quantity)

        # Create density plot
        fig = go.Figure()

        range_config = [
            ('1만골드 이하', '#FF6B6B'),
            ('2만골드 이하', '#4ECDC4'),
            ('3만골드 이하', '#45B7D1'),
            ('5만골드 이하', '#96CEB4'),
            ('5만골드 초과', '#FFEAA7')
        ]

        for category, color in range_config:
            category_data = trades_df[trades_df['category'] == category]['gold_per_coin']

            if not category_data.empty:
                # Create histogram with KDE overlay
                fig.add_trace(
                    go.Violin(
                        y=category_data,
                        name=category,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=color,
                        opacity=0.6,
                        x0=category
                    )
                )

        fig.update_layout(
            title=f"GPC Distribution by Quantity Range (Latest: {latest_date})",
            yaxis_title="Gold per Coin",
            xaxis_title="Quantity Range",
            height=600,
            showlegend=True,
            violinmode='group'
        )

        return fig

    def create_gpc_bar_chart(self) -> go.Figure:
        """Create scatter plot showing GPC by gold quantity for the latest date"""
        # Get the latest snapshot
        df = self.db.get_time_series_data()

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Get the most recent data point
        latest_row = df.iloc[-1]
        latest_date = latest_row['timestamp'].strftime('%Y-%m-%d')

        # Get all trades from the latest snapshot
        conn = sqlite3.connect(self.db.db_path)
        query = '''
            SELECT seller, quantity, gold_per_coin
            FROM trades
            WHERE source_file = ?
            ORDER BY quantity ASC, gold_per_coin ASC
        '''
        trades_df = pd.read_sql_query(query, conn, params=(latest_row['source_file'],))
        conn.close()

        if trades_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Create scatter plot
        fig = go.Figure()

        # Add scatter plot with color gradient based on GPC
        fig.add_trace(
            go.Scatter(
                x=trades_df['quantity'],
                y=trades_df['gold_per_coin'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=trades_df['gold_per_coin'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="GPC"),
                    line=dict(width=1, color='white')
                ),
                text=trades_df['seller'],
                hovertemplate='<b>판매자: %{text}</b><br>' +
                             '골드: %{x:,}<br>' +
                             'GPC: %{y:.1f}<extra></extra>'
            )
        )

        fig.update_layout(
            title=f"Gold Quantity vs GPC (Latest: {latest_date})",
            xaxis_title="Gold Quantity",
            yaxis_title="Gold per Coin (GPC)",
            height=600,
            showlegend=False,
            hovermode='closest'
        )

        # Update axes to start from low to high
        fig.update_xaxes(range=[trades_df['quantity'].min() * 0.95, trades_df['quantity'].max() * 1.05])
        fig.update_yaxes(range=[trades_df['gold_per_coin'].min() * 0.95, trades_df['gold_per_coin'].max() * 1.05])

        return fig

    def create_optimal_selling_strategy(self) -> go.Figure:
        """Analyze optimal selling strategy by finding best GPC for each coin price"""
        # Get the latest snapshot
        df = self.db.get_time_series_data()

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Get the most recent data point
        latest_row = df.iloc[-1]
        latest_date = latest_row['timestamp'].strftime('%Y-%m-%d')

        # Get all trades from the latest snapshot
        conn = sqlite3.connect(self.db.db_path)
        query = '''
            SELECT seller, quantity, gold_per_coin, price
            FROM trades
            WHERE source_file = ?
            ORDER BY price ASC, gold_per_coin DESC
        '''
        trades_df = pd.read_sql_query(query, conn, params=(latest_row['source_file'],))
        conn.close()

        if trades_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Group by coin price and find the best (highest GPC) for each price
        price_leaders = trades_df.loc[trades_df.groupby('price')['gold_per_coin'].idxmax()]

        # Sort by price for readability
        price_leaders = price_leaders.sort_values('price')

        # Create bar chart
        fig = go.Figure()

        # Add bars with color gradient based on GPC
        fig.add_trace(
            go.Bar(
                x=price_leaders['price'],
                y=price_leaders['gold_per_coin'],
                text=price_leaders['gold_per_coin'].apply(lambda x: f'{x:.1f}'),
                textposition='outside',
                marker=dict(
                    color=price_leaders['gold_per_coin'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="GPC"),
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>판매자: %{customdata[0]}</b><br>' +
                             '가격: %{x:,} 코인<br>' +
                             '골드: %{customdata[1]:,}<br>' +
                             'GPC: %{y:.1f}<extra></extra>',
                customdata=price_leaders[['seller', 'quantity']].values
            )
        )

        fig.update_layout(
            title=f"💰 코인 가격별 최고 GPC (Latest: {latest_date})",
            xaxis_title="Price (Coins)",
            yaxis_title="Gold per Coin (GPC)",
            height=700,
            showlegend=False
        )

        # Format x-axis to show coin prices
        fig.update_xaxes(
            tickmode='array',
            tickvals=price_leaders['price'],
            ticktext=[f"{int(p)}" for p in price_leaders['price']]
        )

        return fig

    def create_net_profit_analysis(self) -> go.Figure:
        """Analyze seller's real CPG (Coins Per Gold) after service fee deduction"""
        import math

        # Get the latest snapshot
        df = self.db.get_time_series_data()

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Get the most recent data point
        latest_row = df.iloc[-1]
        latest_date = latest_row['timestamp'].strftime('%Y-%m-%d')

        # Get all trades from the latest snapshot
        conn = sqlite3.connect(self.db.db_path)
        query = '''
            SELECT seller, quantity, gold_per_coin, price
            FROM trades
            WHERE source_file = ?
            ORDER BY price ASC
        '''
        trades_df = pd.read_sql_query(query, conn, params=(latest_row['source_file'],))
        conn.close()

        if trades_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Calculate service fee for each trade
        # Fee formula: ceiling(price / 10)
        trades_df['service_fee'] = trades_df['price'].apply(lambda p: math.ceil(p / 10))

        # Calculate net profit (actual coins received after fee)
        # Seller receives: price - service_fee
        trades_df['net_coins'] = trades_df['price'] - trades_df['service_fee']

        # Calculate CPG (Coins Per Gold) after fee
        # CPG = net_coins / quantity (실제 받는 코인 / 판매한 골드량)
        trades_df['cpg_after_fee'] = trades_df['net_coins'] / trades_df['quantity']

        # Group by coin price and find the most competitive seller (highest GPC = most gold per coin)
        # This is the seller buyers will choose (best offer for buyers)
        price_leaders = trades_df.loc[trades_df.groupby('price')['gold_per_coin'].idxmax()]

        # Sort by price for readability
        price_leaders = price_leaders.sort_values('price')

        # Create bar chart
        fig = go.Figure()

        # Add bars - higher CPG is better (more coins received per gold sold)
        fig.add_trace(
            go.Bar(
                x=price_leaders['price'],
                y=price_leaders['cpg_after_fee'],
                text=price_leaders['cpg_after_fee'].apply(lambda x: f'{x:.5f}'),
                textposition='outside',
                marker=dict(
                    color=price_leaders['cpg_after_fee'],
                    colorscale='Viridis',  # Higher is better (green)
                    showscale=True,
                    colorbar=dict(title="CPG"),
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>판매자: %{customdata[0]}</b><br>' +
                             '표시 가격: %{x:,} 코인<br>' +
                             '수수료: %{customdata[1]:,} 코인<br>' +
                             '실제 수익: %{customdata[2]:,} 코인<br>' +
                             '판매 골드: %{customdata[3]:,}<br>' +
                             'CPG (수수료 후): %{y:.5f}<extra></extra>',
                customdata=price_leaders[['seller', 'service_fee', 'net_coins', 'quantity']].values
            )
        )

        fig.update_layout(
            title=f"💸 코인 가격별 최고 CPG (수수료 공제 후) - Latest: {latest_date}",
            xaxis_title="Price (Coins)",
            yaxis_title="Coins Per Gold (높을수록 좋음)",
            height=700,
            showlegend=False
        )

        # Format x-axis to show coin prices
        fig.update_xaxes(
            tickmode='array',
            tickvals=price_leaders['price'],
            ticktext=[f"{int(p)}<br>(-{int(row['service_fee'])})"
                     for p, row in zip(price_leaders['price'],
                                      price_leaders.to_dict('records'))]
        )

        return fig

# Gradio interface functions
def process_text_input(analyzer: GoldTradeAnalyzer, text_input: str, source_file: str = "manual_input") -> str:
    """Process text input and store in database"""
    if not text_input.strip():
        return "[ERROR] 텍스트가 비어있습니다."

    analyzer.current_trades = analyzer.parse_gold_trades_from_text(text_input)
    analyzer.current_timestamp = datetime.now()

    if analyzer.current_trades:
        # Store in database
        analyzer.db.insert_trades(
            analyzer.current_trades,
            analyzer.current_timestamp,
            source_file
        )

        # Store market snapshot
        snapshot = analyzer.calculate_market_snapshot(analyzer.current_trades)
        analyzer.db.insert_market_snapshot(
            snapshot,
            analyzer.current_timestamp,
            source_file
        )

        # Provide detailed feedback
        best_trade = analyzer.current_trades[0]
        result = f"[SUCCESS] 성공적으로 {len(analyzer.current_trades)}개의 거래를 처리했습니다!\n\n"
        result += f"최고 가성비: {best_trade['seller']} - {best_trade['gpc']:.1f} 골드/코인\n"
        result += f"총 거래량: {sum(t['quantity'] for t in analyzer.current_trades):,} 골드\n"
        result += f"판매자 수: {len(set(t['seller'] for t in analyzer.current_trades))}명\n"
        result += f"처리 시간: {analyzer.current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

        return result
    else:
        # Provide debugging information
        lines = text_input.strip().split('\n')
        gold_lines = [i for i, line in enumerate(lines) if 'gold' in line.lower()]

        result = "[ERROR] 유효한 거래 데이터를 찾을 수 없습니다.\n\n"
        result += f"총 라인 수: {len(lines)}\n"
        result += f"'gold' 포함 라인: {len(gold_lines)}개\n\n"

        if gold_lines:
            result += "발견된 골드 라인들:\n"
            for i in gold_lines[:5]:  # Show first 5 gold lines
                result += f"Line {i+1}: {lines[i][:50]}...\n"

            result += "\n[확인사항]\n"
            result += "- 골드 라인 다음에 Duration, Seller, Price 정보가 있는지 확인\n"
            result += "- 형식: 'Long Seller 150 coins' 또는 탭으로 구분된 형식\n"
        else:
            result += "[TIP] 'gold'가 포함된 라인이 없습니다. 데이터 형식을 확인해주세요."

        return result

def get_trades_table(analyzer: GoldTradeAnalyzer) -> pd.DataFrame:
    """Get current trades table from latest database snapshot"""
    # Get the latest snapshot from database
    df = analyzer.db.get_time_series_data()

    if df.empty:
        return pd.DataFrame()

    # Get the most recent data point
    latest_row = df.iloc[-1]

    # Get all trades from the latest snapshot
    conn = sqlite3.connect(analyzer.db.db_path)
    query = '''
        SELECT seller, quantity, price, gold_per_coin, duration
        FROM trades
        WHERE source_file = ?
        ORDER BY gold_per_coin DESC
    '''
    trades_df = pd.read_sql_query(query, conn, params=(latest_row['source_file'],))
    conn.close()

    if trades_df.empty:
        return pd.DataFrame()

    # Rename columns
    trades_df.columns = ['판매자', '총 골드', '가격(코인)', '코인당 골드', '기간']

    # Format numbers with commas
    trades_df['총 골드'] = trades_df['총 골드'].apply(lambda x: f"{x:,}")
    trades_df['가격(코인)'] = trades_df['가격(코인)'].apply(lambda x: f"{x:,}")

    return trades_df

def import_historical_data(analyzer: GoldTradeAnalyzer, force_refresh: bool = False) -> str:
    """Import all historical txt files"""
    result = analyzer.import_txt_files(force_refresh=force_refresh)

    # Add more detailed information
    df = analyzer.db.get_time_series_data()
    if not df.empty:
        total_snapshots = len(df)
        date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} ~ {df['timestamp'].max().strftime('%Y-%m-%d')}"
        best_gpc_filtered_range = f"{df['best_gpc_filtered'].min():.1f} ~ {df['best_gpc_filtered'].max():.1f}"

        detailed_result = f"{result}\n\n[데이터베이스 정보]\n"
        detailed_result += f"- 일별 스냅샷: {total_snapshots}개\n"
        detailed_result += f"- 기간: {date_range}\n"
        detailed_result += f"- 최고 GPC 범위 (>=10k): {best_gpc_filtered_range}\n"
        detailed_result += f"- 총 거래량: {df['total_volume'].sum():,} 골드"

        return detailed_result

    return result

def get_time_series_chart(analyzer: GoldTradeAnalyzer, limit: int = None) -> go.Figure:
    """Get time series chart"""
    return analyzer.create_time_series_chart(limit)

def get_volatility_chart(analyzer: GoldTradeAnalyzer) -> go.Figure:
    """Get volatility chart"""
    return analyzer.create_price_volatility_chart()

def get_seller_performance_chart(analyzer: GoldTradeAnalyzer) -> go.Figure:
    """Get seller performance chart"""
    return analyzer.create_seller_performance_chart()

def get_gpc_density_chart(analyzer: GoldTradeAnalyzer) -> go.Figure:
    """Get GPC density chart"""
    return analyzer.create_gpc_density_chart()

def get_gpc_bar_chart(analyzer: GoldTradeAnalyzer) -> go.Figure:
    """Get GPC bar chart"""
    return analyzer.create_gpc_bar_chart()

def get_optimal_strategy(analyzer: GoldTradeAnalyzer) -> go.Figure:
    """Get optimal selling strategy analysis"""
    return analyzer.create_optimal_selling_strategy()

def get_net_profit_analysis(analyzer: GoldTradeAnalyzer) -> go.Figure:
    """Get net profit analysis after service fees"""
    return analyzer.create_net_profit_analysis()

def save_text_to_file(text_input: str) -> Tuple[str, str]:
    """Save input text to timestamped file and return filename and status"""
    if not text_input.strip():
        return "", "텍스트가 비어있습니다."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gold_data_{timestamp}.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text_input)
        return filename, f"파일이 저장되었습니다: {filename}"
    except Exception as e:
        return "", f"저장 실패: {str(e)}"

def save_and_process_combined(analyzer: GoldTradeAnalyzer, text_input: str) -> str:
    filename, save_status = save_text_to_file(text_input)
    if filename:
        process_status = process_text_input(analyzer, text_input, filename)
        return f"{save_status}\n{process_status}"
    else:
        return save_status

# Initialize the analyzer
analyzer = GoldTradeAnalyzer()

# Automatically import historical data on startup
print("Automatically importing historical data...")
initial_import_status = import_historical_data(analyzer)
print(initial_import_status)

# Create Gradio interface
with gr.Blocks(title="WoW 골드 거래 분석기 Enhanced", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🏆 WoW 골드 거래 분석기 Enhanced")
    gr.Markdown("### 시계열 분석 및 데이터베이스 통합 버전")

    with gr.Tabs():
        # Tab 1: Data Input and Import
        with gr.Tab("📝 데이터 입력 및 저장"):
            gr.Markdown("## 데이터 입력 및 저장")

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 새 데이터 입력")
                    text_input = gr.Textbox(
                        lines=15,
                        placeholder="여기에 골드 거래 데이터를 붙여넣으세요...",
                        label="거래 데이터 텍스트"
                    )

                    with gr.Row():
                        save_process_btn = gr.Button("💾 저장 및 처리", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 상태")
                    status_output = gr.Textbox(label="처리 상태", interactive=False, lines=3)

            # Event handlers
            save_process_btn.click(
                lambda text: save_and_process_combined(analyzer, text),
                inputs=[text_input],
                outputs=[status_output]
            )

        # Tab 2: Analysis
        with gr.Tab("📊 분석"):
            with gr.Tabs(): # Nested tabs for different analysis types
                # Content from original "📈 시계열 분석"
                with gr.Tab("📈 시계열 분석"):
                    gr.Markdown("## 일별 시장 동향 분석")
                    gr.Markdown("**참고**: X축은 파일 순서를 나타냅니다. 같은 날짜에 여러 파일이 있는 경우 최신 파일만 사용됩니다.")

                    with gr.Row():
                        limit_slider = gr.Slider(
                            minimum=5, maximum=100, value=30, step=5,
                            label="표시할 데이터 수", info="최근 몇 개의 데이터를 표시할지 선택"
                        )
                        refresh_timeseries_btn = gr.Button("🔄 차트 업데이트 (파일 재파싱)", variant="primary")

                    timeseries_chart = gr.Plot(label="시계열 분석 차트")

                    refresh_timeseries_btn.click(
                        lambda limit: (import_historical_data(analyzer, force_refresh=True), get_time_series_chart(analyzer, int(limit)))[1],
                        inputs=[limit_slider],
                        outputs=[timeseries_chart]
                    )

                # Content from original "🏆 현재 순위"
                with gr.Tab("🏆 현재 순위"):
                    gr.Markdown("## 최신 데이터 분석 결과")
                    gr.Markdown("**참고**: 데이터베이스에 저장된 최신 스냅샷의 거래 내역을 표시합니다.")

                    refresh_table_btn = gr.Button("🔄 테이블 새로고침")
                    trades_table = gr.Dataframe(
                        headers=["판매자", "총 골드", "가격(코인)", "코인당 골드", "기간"],
                        label="가성비 순위",
                        value=get_trades_table(analyzer)
                    )

                    refresh_table_btn.click(
                        lambda: get_trades_table(analyzer),
                        outputs=[trades_table]
                    )

                # NEW: GPC Density Plot
                with gr.Tab("📊 GPC 분포도"):
                    gr.Markdown("## 최신 데이터의 금액구간별 GPC 분포")
                    gr.Markdown("**참고**: 바이올린 플롯으로 각 금액 구간의 GPC 분포를 시각화합니다.")

                    refresh_density_btn = gr.Button("🔄 분포도 업데이트", variant="primary")
                    density_chart = gr.Plot(label="GPC 분포도")

                    refresh_density_btn.click(
                        lambda: get_gpc_density_chart(analyzer),
                        outputs=[density_chart]
                    )

                # NEW: GPC Bar Chart
                with gr.Tab("📊 골드별 GPC"):
                    gr.Markdown("## 최신 데이터의 골드 수량별 GPC")
                    gr.Markdown("**참고**: 각 거래의 골드 수량과 GPC를 산점도로 표시합니다.")

                    refresh_bar_btn = gr.Button("🔄 차트 업데이트", variant="primary")
                    bar_chart = gr.Plot(label="골드별 GPC 차트")

                    refresh_bar_btn.click(
                        lambda: get_gpc_bar_chart(analyzer),
                        outputs=[bar_chart]
                    )

                # NEW: Optimal Selling Strategy
                with gr.Tab("🎯 최적 판매 전략"):
                    gr.Markdown("## 💰 코인 가격별 최고 GPC")
                    gr.Markdown("**핵심**: 각 코인 가격에서 GPC가 가장 높은 판매자가 1등입니다!")
                    gr.Markdown("- 막대 높이: 해당 코인 가격에서의 최고 GPC (Gold per Coin)")
                    gr.Markdown("- 막대 색상: GPC 값 (진한 색 = 높은 효율)")

                    refresh_strategy_btn = gr.Button("🔄 차트 업데이트", variant="primary")
                    strategy_chart = gr.Plot(label="코인 가격별 최고 GPC")

                    refresh_strategy_btn.click(
                        lambda: get_optimal_strategy(analyzer),
                        outputs=[strategy_chart]
                    )

                # NEW: CPG Analysis with Service Fee
                with gr.Tab("💸 수수료 반영 CPG 분석"):
                    gr.Markdown("## 📈 경쟁 판매자의 실제 수익 효율")
                    gr.Markdown("**수수료 구조**: 판매가격 1~14 코인 → 수수료 1코인, 15~24 코인 → 수수료 2코인...")
                    gr.Markdown("**수수료 계산**: 수수료 = ⌈가격 / 10⌉ (올림)")
                    gr.Markdown("")
                    gr.Markdown("**차트 설명**:")
                    gr.Markdown("- **대상**: 각 코인 가격에서 **GPC가 가장 높은** 판매자 (구매자가 선택할 사람)")
                    gr.Markdown("- **X축**: 표시 가격 (코인)")
                    gr.Markdown("- **Y축**: 그 판매자의 실제 CPG = (가격 - 수수료) / 골드량")
                    gr.Markdown("- **예시**: 10코인 가격대에서 A가 30,000골드로 1등 → A의 CPG = (10-1)/30,000 = 0.0003")
                    gr.Markdown("- **의미**: 각 가격대에서 이기려면 이 CPG보다 높거나 같아야 함")

                    refresh_profit_btn = gr.Button("🔄 차트 업데이트", variant="primary")
                    profit_chart = gr.Plot(label="수수료 반영 CPG 분석")

                    refresh_profit_btn.click(
                        lambda: get_net_profit_analysis(analyzer),
                        outputs=[profit_chart]
                    )

if __name__ == "__main__":
    app.launch(share=False, inbrowser=True, server_port=7860)