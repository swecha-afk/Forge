import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from app.core.config import settings

def init_mandi_db() -> None:
    os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mandi_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            commodity TEXT NOT NULL,
            state TEXT NOT NULL,
            district TEXT NOT NULL,
            market TEXT NOT NULL,
            modal_price REAL NOT NULL,
            arrival_date TEXT NOT NULL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Pre-seed baseline historical data if empty so live demo never hits a cold-start failure
    cursor.execute("SELECT COUNT(*) FROM mandi_prices")
    if cursor.fetchone()[0] == 0:
        seed_baseline_data(cursor)
        
    conn.commit()
    conn.close()

def seed_baseline_data(cursor: sqlite3.Cursor) -> None:
    base_date = datetime.now()
    sample_crops = [
        ("Potato", "West Bengal", "Medinipur West", "Kharagpur APMC", [1650, 1680, 1700, 1720, 1690, 1710, 1840]),
        ("Tomato", "West Bengal", "Medinipur West", "Kharagpur APMC", [2800, 2750, 2700, 2650, 2600, 2550, 2400]),
        ("Rice", "West Bengal", "Medinipur West", "Kharagpur APMC", [3100, 3120, 3110, 3130, 3125, 3140, 3150]),
    ]
    for commodity, state, district, market, prices in sample_crops:
        for i, price in enumerate(prices):
            day_str = (base_date - timedelta(days=7 - i)).strftime("%Y-%m-%d")
            cursor.execute("""
                INSERT INTO mandi_prices (commodity, state, district, market, modal_price, arrival_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (commodity.lower(), state.lower(), district.lower(), market, float(price), day_str))

def cache_live_price(commodity: str, state: str, district: str, market: str, modal_price: float, arrival_date: str) -> None:
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # Prevent duplicate live inserts on the same day from skewing the 7-day moving average
    cursor.execute("""
        SELECT id FROM mandi_prices 
        WHERE commodity = ? AND district = ? AND arrival_date = ?
    """, (commodity.lower(), district.lower(), arrival_date))
    
    existing = cursor.fetchone()
    if existing:
        cursor.execute("""
            UPDATE mandi_prices 
            SET modal_price = ?, market = ?, recorded_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (modal_price, market, existing[0]))
    else:
        cursor.execute("""
            INSERT INTO mandi_prices (commodity, state, district, market, modal_price, arrival_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (commodity.lower(), state.lower(), district.lower(), market, modal_price, arrival_date))
        
    conn.commit()
    conn.close()

def get_cached_latest_and_avg(commodity: str, district: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT market, modal_price, arrival_date 
        FROM mandi_prices 
        WHERE commodity = ? AND district = ?
        ORDER BY id DESC LIMIT 7
    """, (commodity.lower(), district.lower()))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return None
        
    latest_market, latest_price, latest_date = rows[0]
    avg_price = sum(r[1] for r in rows) / len(rows)
    
    return {
        "market": latest_market,
        "modal_price": latest_price,
        "arrival_date": latest_date,
        "seven_day_avg": round(avg_price, 2)
    }