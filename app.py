import streamlit as st
import yfinance as yf
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="AI 주가 일기예보", page_icon="📈", layout="centered")

def get_smart_ticker(user_input):
    user_input = user_input.upper().strip()
    if user_input.isdigit() and len(user_input) == 6:
        if not yf.Ticker(user_input + ".KS").history(period="1d").empty: return user_input + ".KS"
        else: return user_input + ".KQ"
    futures_map = {"NQ": "NQ=F", "ES": "ES=F", "YM": "YM=F", "CL": "CL=F", "GC": "GC=F"}
    if user_input in futures_map: return futures_map[user_input]
    if user_input in ["BTC", "ETH"]: return user_input + "-USD"
    return user_input

def add_features(data):
    df = data.copy()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Trend_20'] = (df['Close'] / df['MA20']) - 1
    df['Momentum_10'] = (df['Close'] / df['Close'].shift(10)) - 1
    candle_range = df['High'] - df['Low'] + 0.0001
    df['Candle_Pullback'] = (df['Close'] - df['Low']) / candle_range  
    df['Candle_Rejection'] = (df['High'] - df['Close']) / candle_range 
    df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Spike'] = df['Volume'] / (df['Vol_MA5'] + 0.0001)
    
    def get_stoch(n):
        L = df['Low'].rolling(window=n).min()
        H = df['High'].rolling(window=n).max()
        return (df['Close'] - L) / (H - L + 0.0001) * 100
    df['Stoch_5'] = get_stoch(5)
    df['Stoch_14'] = get_stoch(14)
    df['Stoch_20'] = get_stoch(20)
    
    def get_rsi(n):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/n, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/n, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    df['RSI_5'] = get_rsi(5)
    df['RSI_14'] = get_rsi(14)
    df['RSI_21'] = get_rsi(21)
    
    return df.dropna()

def predict_direction(df, shift_period, target_type='Close'):
    if target_type == 'Close':
        df['Target'] = (df['Close'].shift(-shift_period) > df['Close']).astype(int)
    else: 
        df['Target'] = (df['Open'].shift(-shift_period) > df['Close']).astype(int)
        
    df = df.dropna()
    if len(df) < 50: return 0.5 
    
    features = ['Trend_20', 'Momentum_10', 'Candle_Pullback', 'Candle_Rejection', 
                'Volume_Spike', 'Stoch_5', 'Stoch_14', 'Stoch_20', 'RSI_5', 'RSI_14', 'RSI_21']
    
    # ⚡ 속도 최적화: 300명의 스나이퍼 특공대로 압축! (n_estimators=300)
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
    
    train_data = df.iloc[:-1]
    latest_data = df.iloc[[-1]]
    
    model.fit(train_data[features], train_data['Target'])
    prob = model.predict_proba(latest_data[features])[0][1] 
    return prob

def get_weather(prob):
    if prob >= 0.65: return "☀️ 맑음"
    elif prob >= 0.55: return "⛅ 약간"
    elif prob >= 0.45: return "☁️ 흐림"
    elif prob >= 0.35: return "🌧️ 비"
    else: return "⛈️ 폭우"

# --- 화면 UI 구성 ---
st.title("🚀 AI 실시간 주가 예보")
st.markdown("정예 300명의 AI 특공대가 11개 지표를 실시간으로 빠르게 분석합니다.")

raw_input = st.text_input("🎯 종목코드 입력 (예: TQQQ, 005930, NQ, BTC)", "")

if st.button("초고속 분석 시작 ⚡"):
    if not raw_input:
        st.warning("종목 코드를 입력해주세요!")
    else:
        ticker = get_smart_ticker(raw_input)
        
        with st.spinner(f"⏳ [{ticker}] AI 특공대 분석 중..."):
            try:
                data_5m = yf.Ticker(ticker).history(period="60d", interval="5m")
                data_1d = yf.Ticker(ticker).history(period="5y", interval="1d")
                
                if data_1d.empty or data_5m.empty:
                    st.error(f"❌ '{ticker}' 데이터를 찾을 수 없습니다. 다시 입력해주세요.")
                else:
                    df_5m = add_features(data_5m)
                    df_1d = add_features(data_1d)

                    probs = {
                        "🕒 5분후": predict_direction(df_5m, 1),
                        "🕒 15분후": predict_direction(df_5m, 3),
                        "🕒 30분후": predict_direction(df_5m, 6),
                        "🕒 60분후": predict_direction(df_5m, 12),
                        "🌅 낼시가": predict_direction(df_1d, 1, target_type='Open'),
                        "🌇 낼종가": predict_direction(df_1d, 1)
                    }

                    results = []
                    for time_label, prob in probs.items():
                        weather = get_weather(prob)
                        direction = "🔴 상승" if prob >= 0.5 else "🔵 하락"
                        display_prob = prob if prob >= 0.5 else (1 - prob)
                        
                        results.append({
                            "시간": time_label,
                            "날씨": weather,
                            "예측 방향": f"{direction} ({display_prob * 100:.1f}%)"
                        })
                    
                    df_results = pd.DataFrame(results)
                    st.success(f"📈 [{ticker}] 고속 분석 완료!")
                    st.table(df_results)
                    
            except Exception as e:
                st.error(f"❌ 분석 중 오류가 발생했습니다: {e}")
