"""
AI BUDGET TRACKER – FINAL PRODUCTION VERSION
BCA Major Project | Presentation Ready: April 21, 2026

Features:
- Real Tesseract OCR receipt scanning (pytesseract + PIL)
- Mobile-responsive layout (collapsed sidebar, narrow login columns)
- Cloud-Native Firebase with Streamlit Secrets support
- Dual Linear Regression forecast (Income & Expense)
- Isolation Forest anomaly detection
- 50/30/20 AI Advisory Engine with goals tracking
- 90-day seeder + Bank sync simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import hashlib
import secrets
from datetime import datetime, timedelta
from PIL import Image
import pytesseract
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth as firebase_auth
    FIREBASE_SDK = True
except ImportError:
    FIREBASE_SDK = False

CURRENCY = "₹"

# ─── TESSERACT CONFIG ──────────────────────────────────────────────────────────
# Set path for Windows; on Linux/Cloud use 'tesseract' in PATH
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ─── RESPONSIVE THEME ──────────────────────────────────────────────────────────
THEME_CSS = """
<style>
    html, body, [class*="st-"] { font-family: 'Inter', -apple-system, sans-serif; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d44;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4);
    }
    div[data-testid="stMetricValue"] { color: #00d4ff !important; font-weight: 700; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
        border-right: 1px solid #2d2d44;
    }
    h1, h2, h3 { color: #e0e6f0; }
    h1 {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1e1e2f;
        border-radius: 10px 10px 0 0;
        padding: 12px 20px;
        color: #b0b8c8;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff22, #00ff8822);
        border: 1px solid #00d4ff44;
        color: #00d4ff !important;
        font-weight: 600;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid #2d2d44;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        border-radius: 8px;
    }
    /* Mobile centering for auth */
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        background: #1a1a2e;
        border-radius: 16px;
        border: 1px solid #2d2d44;
    }
    /* 1. COMPLETELY WIPE THE BROKEN TEXT */
    button[kind="header"] {
        color: transparent !important;
        background: transparent !important;
        overflow: hidden !important;
        width: 50px !important;
    }
    /* 2. POSITION THE NEW MENU ICON CLEANLY */
    button[kind="header"]:after {
        content: '☰'; 
        visibility: visible;
        display: block;
        position: absolute;
        color: #00d4ff !important; 
        font-size: 30px;
        top: 8px;
        left: 12px;
        font-weight: bold;
    }
    /* 3. EXPAND CAMERA & FORCE HIGH RESOLUTION */
    div[data-testid="stCameraInput"] {
        width: 100% !important;
        padding: 0 !important;
    }
    /* This forces the camera container to be much larger on mobile */
    div[data-testid="stCameraInput"] > div {
        width: 100% !important;
        min-height: 70vh !important; /* Uses 70% of the phone screen height */
    }
    /* Force the video feed to fill the space without blurring */
    video {
        object-fit: cover !important;
        width: 100% !important;
        height: 70vh !important;
        min-height: 500px !important;
        border-radius: 15px;
        border: 2px solid #2d2d44;
    }
</style>
"""

# ─── CATEGORY TAXONOMY ─────────────────────────────────────────────────────────
INCOME_CATS = ["Salary", "Freelance", "Bonus", "Investment", "Other"]
EXPENSE_CATS = ["Food", "Transport", "Shopping", "Bills", "Rent", "Other"]

# Vendor keyword → category mapping for auto-categorization after OCR
VENDOR_MAP = {
    "zomato": "Food", "swiggy": "Food", "domino": "Food", "mcdonalds": "Food",
    "big basket": "Food", "blinkit": "Food", "zepto": "Food",
    "uber": "Transport", "ola": "Transport", "rapido": "Transport",
    "amazon": "Shopping", "flipkart": "Shopping", "myntra": "Shopping",
    "netflix": "Entertainment", "prime": "Entertainment", "hotstar": "Entertainment",
    "electricity": "Bills", "discom": "Bills", " water": "Bills", "internet": "Bills",
    "mobile": "Bills", "recharge": "Bills",
    "rent": "Rent", "lease": "Rent",
    "gym": "Health", "pharmacy": "Health", "apollo": "Health",
}

SEED_TEMPLATES = [
    ("Zomato order", "Food", 150, 400),
    ("Swiggy delivery", "Food", 200, 500),
    ("Big Basket groceries", "Food", 500, 1500),
    ("Uber trip", "Transport", 100, 300),
    ("Ola ride", "Transport", 100, 250),
    ("Metro recharge", "Transport", 50, 100),
    ("Amazon purchase", "Shopping", 500, 3000),
    ("Myntra fashion", "Shopping", 500, 2500),
    ("Netflix subscription", "Entertainment", 199, 499),
    ("Movie ticket", "Entertainment", 200, 400),
    ("Electricity bill", "Bills", 800, 2000),
    ("Internet bill", "Bills", 600, 1000),
    ("Mobile recharge", "Bills", 199, 999),
    ("Rent payment", "Rent", 15000, 25000),
    ("Gym membership", "Health", 500, 1500),
    ("Pharmacy medicines", "Health", 100, 800),
    ("Doctor consultation", "Health", 300, 1000),
]

INCOME_TEMPLATES = [
    ("Monthly Salary", "Salary", 48000, 52000),
    ("Freelance Project", "Freelance", 5000, 15000),
    ("Stock Dividend", "Investment", 1000, 5000),
    ("Performance Bonus", "Bonus", 10000, 20000),
]

firebase_ok = False
model_cache = {}

# ─── FIREBASE WITH SECRETS SUPPORT ─────────────────────────────────────────────
def init_firebase():
    """Initialize Firebase for local (serviceAccountKey.json) or Streamlit Cloud (secrets)"""
    global firebase_ok
    if not FIREBASE_SDK:
        st.error("❌ Firebase Admin SDK missing. Install: `pip install firebase-admin`")
        st.stop()
    if firebase_ok:
        return True
    try:
        if len(firebase_admin._apps) > 0:
            firebase_ok = True
            return True
    except:
        pass
    # Try Streamlit secrets first (Cloud)
    if hasattr(st, 'secrets') and 'firebase' in st.secrets:
        try:
            cred_dict = dict(st.secrets['firebase'])
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            firebase_ok = True
            return True
        except Exception as e:
            st.error(f"🔴 Firebase secrets error: {e}")
            st.stop()
    # Fallback to local file
    cred_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
    if not os.path.exists(cred_path):
        st.error("🔴 CRITICAL: `serviceAccountKey.json` not found. "
                 "Download from Firebase Console → Project Settings → Service Accounts.")
        st.stop()
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        firebase_ok = True
        return True
    except ValueError as ve:
        if "already exists" in str(ve).lower():
            firebase_ok = True
            return True
        raise ve
    except Exception as e:
        st.error(f"🔴 Firebase init failed: {e}")
        st.stop()

# ─── OCR ENGINE ─────────────────────────────────────────────────────────────────
def perform_ocr(image: Image) -> dict:
    """
    Extract text from receipt image using Tesseract OCR.
    Returns:
        dict: { 'text': str, 'amount': float|None, 'vendor': str|None }
    """
    result = {'text': '', 'amount': None, 'vendor': None}
    try:
        # Preprocess: grayscale + upscale
        img = image.convert('L')
        text = pytesseract.image_to_string(img, config='--psm 6')
        result['text'] = text.strip()
        # Find largest decimal number (amount)
        numbers = re.findall(r'(\d+\.\d{2,})', text)
        if numbers:
            amounts = [float(n) for n in numbers]
            result['amount'] = max(amounts)
        # Find vendor keyword
        text_lower = text.lower()
        for keyword, cat in VENDOR_MAP.items():
            if keyword in text_lower:
                result['vendor'] = keyword.title()
                result['category'] = cat
                break
        if not result.get('vendor'):
            result['vendor'] = 'Unknown'
    except Exception as e:
        st.error(f"OCR error: {e}")
    return result

# ─── UTILITIES ──────────────────────────────────────────────────────────────────
def fmt(amount):
    if amount is None:
        return f"{CURRENCY}0"
    sign = "-" if amount < 0 else ""
    amt = f"{abs(amount):.2f}"
    int_p, dec_p = amt.split('.')
    if len(int_p) > 3:
        last3 = int_p[-3:]
        rem = int_p[:-3]
        parts = [rem[max(0,i-2):i] for i in range(len(rem), 0, -2)][::-1]
        int_p = ','.join(filter(None, parts)) + ',' + last3 if parts else last3
    return f"{CURRENCY}{sign}{int_p}.{dec_p}"

def get_db():
    return firestore.client() if (FIREBASE_SDK and firebase_ok) else None

def hash_pwd(pwd, salt=None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac('sha256', pwd.encode(), salt.encode(), 100000).hex()
    return salt + h

def verify_pwd(pwd, stored):
    if len(stored) < 32:
        return False
    salt, pwd_hash = stored[:32], stored[32:]
    return hashlib.pbkdf2_hmac('sha256', pwd.encode(), salt.encode(), 100000).hex() == pwd_hash

def store_credentials(uid, email, pwd_hash):
    db = get_db()
    if not db:
        return False
    try:
        db.collection('users').document(uid).collection('_meta').document('auth').set({
            'email': email, 'password_hash': pwd_hash, 'created_at': firestore.SERVER_TIMESTAMP
        })
        return True
    except:
        return False

def verify_credentials(email, password):
    db = get_db()
    if not db:
        return None
    try:
        user = firebase_auth.get_user_by_email(email)
        doc = db.collection('users').document(user.uid).collection('_meta').document('auth').get()
        if doc.exists and verify_pwd(password, doc.to_dict().get('password_hash', '')):
            return user.uid
    except:
        pass
    return None

def save_prefs(uid, goal_n, goal_a, food_l):
    db = get_db()
    if not db:
        return False
    try:
        db.collection('preferences').document(uid).set({
            'goal_name': goal_n, 'goal_amount': float(goal_a), 'food_limit': float(food_l),
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        return True
    except:
        return False

def load_prefs(uid):
    db = get_db()
    if not db:
        return None
    try:
        doc = db.collection('preferences').document(uid).get()
        return doc.to_dict() if doc.exists else None
    except:
        return None

# ─── TRANSACTION LAYER ─────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_txns(uid):
    db = get_db()
    if not db:
        st.error("☁️ Firestore offline")
        st.stop()
    try:
        docs = (db.collection('users').document(uid)
                .collection('transactions')
                .order_by('date', direction=firestore.Query.DESCENDING)
                .stream())
        rows = []
        for d in docs:
            r = d.to_dict()
            r['id'] = d.id
            rows.append(r)
        if rows:
            df = pd.DataFrame(rows)
            return df[['id','date','amount','description','category','type']]
        return pd.DataFrame(columns=['id','date','amount','description','category','type'])
    except Exception as e:
        st.error(f"Fetch error: {e}")
        return pd.DataFrame(columns=['id','date','amount','description','category','type'])

def save_txn(date, amount, category, ttype, description, uid):
    db = get_db()
    if not db:
        st.error("☁️ Firestore offline")
        st.stop()
    try:
        ref = db.collection('users').document(uid).collection('transactions').document()
        ref.set({
            'date': str(date), 'amount': float(amount), 'category': str(category),
            'type': str(ttype), 'description': str(description),
            'created_at': firestore.SERVER_TIMESTAMP
        })
        return ref.id
    except Exception as e:
        st.error(f"Save failed: {e}")
        return None

def delete_txn(doc_id, uid):
    db = get_db()
    if not db:
        st.error("☁️ Firestore offline")
        st.stop()
    try:
        db.collection('users').document(uid).collection('transactions').document(doc_id).delete()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False

# ─── ML ENGINE ──────────────────────────────────────────────────────────────────
@st.cache_resource
def train_cat_model(df):
    if df.empty or len(df) < 5:
        return None
    try:
        seed = pd.DataFrame([
            ("coffee","Food"), ("lunch","Food"), ("dinner","Food"), ("groceries","Food"),
            ("uber","Transport"), ("ola","Transport"), ("metro","Transport"), ("flight","Transport"),
            ("movie","Entertainment"), ("netflix","Entertainment"), ("prime","Entertainment"),
            ("electricity","Bills"), ("phone bill","Bills"), ("internet","Bills"), ("rent","Rent"),
            ("salary","Salary"), ("freelance","Freelance"), ("bonus","Bonus"), ("investment","Investment"),
            ("shopping","Shopping"), ("gym","Health"), ("medicine","Health"), ("doctor","Health"),
        ], columns=['description','category'])
        combined = pd.concat([seed, df[['description','category']]], ignore_index=True)
        vec = CountVectorizer(ngram_range=(1,2))
        X = vec.fit_transform(combined['description'])
        y = combined['category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        conf = np.mean(np.max(clf.predict_proba(X_test), axis=1))
        model_cache.update({'vectorizer': vec, 'classifier': clf, 'accuracy': acc, 'confidence': conf})
        return acc
    except:
        return None

def predict_category(desc):
    if 'vectorizer' not in model_cache or not desc:
        return "Uncategorized", 0.0
    try:
        vec, clf = model_cache['vectorizer'], model_cache['classifier']
        pred = clf.predict(vec.transform([desc]))[0]
        conf = np.max(clf.predict_proba(vec.transform([desc])))
        return pred, conf
    except:
        return "Uncategorized", 0.0

def detect_anomalies(df):
    if df.empty or len(df) < 5:
        return pd.DataFrame()
    try:
        iso = IsolationForest(contamination=0.1, random_state=42)
        df = df.copy()
        df['anomaly'] = iso.fit_predict(df[['amount']].values)
        return df[df['anomaly'] == -1]
    except:
        return pd.DataFrame()

def forecast_next_30(df):
    if df.empty or len(df) < 2:
        return 0.0, 0.0
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    inc_f, exp_f = 0.0, 0.0
    for ttype in ['Income', 'Expense']:
        sub = df[df['type'] == ttype]
        if len(sub) >= 2:
            monthly = sub.groupby(sub['date'].dt.to_period('M'))['amount'].sum().reset_index()
            if len(monthly) >= 2:
                monthly['idx'] = np.arange(len(monthly))
                reg = LinearRegression().fit(monthly[['idx']], monthly['amount'])
                forecast_val = max(0, reg.predict([[len(monthly)]])[0])
                if ttype == 'Income':
                    inc_f = forecast_val
                else:
                    exp_f = forecast_val
    return inc_f, exp_f

# ─── ADVISORY ENGINE ────────────────────────────────────────────────────────────
def gen_advisory(df, budget, goal_n="", goal_a=0, inc_f=0, exp_f=0):
    tips = []
    if df.empty:
        return ["📝 Add transactions to activate AI financial advisor."]

    exp_df = df[df['type']=='Expense']
    inc_df = df[df['type']=='Income']
    total_exp = exp_df['amount'].sum() if not exp_df.empty else 0
    total_inc = inc_df['amount'].sum() if not inc_df.empty else 0

    # 50/30/20 rule
    if total_inc > 0:
        needs_ratio = total_exp / total_inc
        if needs_ratio > 0.8:
            tips.append(f"⚠️ Expenses consume {needs_ratio:.0%} of income. Aim for 50% needs, 30% wants, 20% savings.")
        elif needs_ratio < 0.4:
            tips.append(f"✅ Expenses only {needs_ratio:.0%} of income. Great savings potential!")

    # Top category alert
    top_cat = exp_df.groupby('category')['amount'].sum().nlargest(1)
    if not top_cat.empty:
        pct = top_cat.iloc[0] / total_exp if total_exp > 0 else 0
        if pct > 0.40:
            tips.append(f"💡 '{top_cat.index[0]}' is {pct:.0%} of spending. Review for optimization.")

    # Budget health
    if total_exp > budget:
        tips.append(f"🚨 Budget exceeded by {fmt(total_exp - budget)}")
    else:
        tips.append(f"✅ Under budget by {fmt(budget - total_exp)}")

    # Goal projection
    if goal_n and goal_a > 0 and inc_f > 0 and exp_f > 0:
        monthly_saving = inc_f - exp_f
        if monthly_saving > 0:
            months = goal_a / monthly_saving
            years = months / 12
            if months <= 36:
                tips.append(f"🎯 '{goal_n}': {months:.1f} months ({years:.1f} years) to {fmt(goal_a)}")
            else:
                tips.append(f"🎯 '{goal_n}': {months:.1f} months. Increase savings to accelerate.")
        else:
            tips.append(f"🎯 '{goal_n}': Reduce expenses to start saving toward your goal.")

    return tips[:3]

# ─── DEMO DATA ──────────────────────────────────────────────────────────────────
def seed_90_days(uid):
    random = np.random.RandomState(42)
    count = 0
    db = get_db()
    for month in [2, 3, 4]:
        year = 2026
        month_str = f"{year}-{month:02d}"
        # Income
        salary = random.randint(48000, 52000)
        doc = db.collection('users').document(uid).collection('transactions').document()
        doc.set({
            'date': f"{month_str}-01", 'amount': float(salary), 'category': 'Salary',
            'type': 'Income', 'description': 'Monthly Salary', 'created_at': firestore.SERVER_TIMESTAMP
        })
        count += 1
        # Expenses
        for day in range(2, 28):
            if random.random() < 0.7:
                template = random.choice(SEED_TEMPLATES)
                amount = random.randint(template[2], template[3])
                doc = db.collection('users').document(uid).collection('transactions').document()
                doc.set({
                    'date': f"{month_str}-{day:02d}", 'amount': float(amount),
                    'category': template[1], 'type': 'Expense', 'description': template[0],
                    'created_at': firestore.SERVER_TIMESTAMP
                })
                count += 1
        # Extra income
        if random.random() < 0.5:
            freelance = random.choice(INCOME_TEMPLATES)
            amount = random.randint(freelance[2], freelance[3])
            doc = db.collection('users').document(uid).collection('transactions').document()
            doc.set({
                'date': f"{month_str}-15", 'amount': float(amount), 'category': freelance[1],
                'type': 'Income', 'description': freelance[0], 'created_at': firestore.SERVER_TIMESTAMP
            })
            count += 1
    return count

def mock_bank_sync(uid):
    mock = [
        ("2026-04-18", 450, "Food", "Expense", "HDFC UPI - Swiggy"),
        ("2026-04-17", 1200, "Food", "Expense", "HDFC Card - BigBasket"),
        ("2026-04-16", 350, "Transport", "Expense", "HDFC UPI - Uber"),
        ("2026-04-15", 2500, "Entertainment", "Expense", "HDFC Card - BookMyShow"),
        ("2026-04-14", 1800, "Bills", "Expense", "HDFC AutoPay - Electricity"),
        ("2026-04-12", 899, "Bills", "Expense", "ICICI Card - Mobile Recharge"),
    ]
    count = 0
    db = get_db()
    for date, amt, cat, typ, desc in mock:
        doc = db.collection('users').document(uid).collection('transactions').document()
        doc.set({
            'date': date, 'amount': float(amt), 'category': cat, 'type': typ,
            'description': desc, 'created_at': firestore.SERVER_TIMESTAMP
        })
        count += 1
    return count

# ─── STREAMLIT CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Budget Tracker",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="💰"
)
st.markdown(THEME_CSS, unsafe_allow_html=True)

init_firebase()

# Session defaults
DEFAULT_SESSION = {
    'user': None,
    'uid': None,
    'monthly_budget': 10000.0,
    'goal_name': '',
    'goal_amount': 50000.0,
    'food_limit': 5000.0,
    'prefs_loaded': False,
}
for k, v in DEFAULT_SESSION.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── AUTH FLOW (CENTERED FOR MOBILE) ──────────────────────────────────────────
if not st.session_state.user:
    st.title("💰 AI Budget Tracker")
    st.caption("Cloud-Native Financial Intelligence | BCA Major Project 2026")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auth_tab1, auth_tab2 = st.tabs(["🔐 Login", "✍️ Register"])
        with auth_tab1:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="your@email.com")
                pwd = st.text_input("Password", type="password")
                login_btn = st.form_submit_button("Login", use_container_width=True)
                if login_btn:
                    if not email or not pwd:
                        st.error("Enter email & password")
                    else:
                        uid = verify_credentials(email, pwd)
                        if uid:
                            st.session_state.user = {'email': email}
                            st.session_state.uid = uid
                            st.session_state.prefs_loaded = False
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
        with auth_tab2:
            with st.form("signup_form"):
                new_email = st.text_input("Email", placeholder="your@email.com")
                new_pwd = st.text_input("Password", type="password", help="Min 6 characters")
                confirm_pwd = st.text_input("Confirm Password", type="password")
                signup_btn = st.form_submit_button("Create Account", use_container_width=True)
                if signup_btn:
                    if new_pwd != confirm_pwd:
                        st.error("Passwords don't match")
                    elif len(new_pwd) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        try:
                            user = firebase_auth.create_user(email=new_email, password=new_pwd)
                            store_credentials(user.uid, new_email, hash_pwd(new_pwd))
                            st.session_state.user = {'email': new_email}
                            st.session_state.uid = user.uid
                            st.session_state.prefs_loaded = False
                            st.success("Account created! Please login.")
                            st.rerun()
                        except firebase_auth.EmailAlreadyExistsError:
                            st.error("Email already registered")
                        except Exception as e:
                            st.error(f"Signup error: {e}")
    st.stop()

uid = st.session_state.uid
user_email = st.session_state.user.get('email', 'User')

# ─── LOAD PREFERENCES ──────────────────────────────────────────────────────────
if not st.session_state.prefs_loaded:
    prefs = load_prefs(uid)
    if prefs:
        st.session_state.goal_name = prefs.get('goal_name', '')
        st.session_state.goal_amount = prefs.get('goal_amount', 50000.0)
        st.session_state.food_limit = prefs.get('food_limit', 5000.0)
    st.session_state.prefs_loaded = True

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"☁️ {user_email}")
    st.markdown("---")

    with st.expander("🛠️ Developer Tools", expanded=False):
        if st.button("🏦 HDFC/ICICI Bank Sync", use_container_width=True):
            n = mock_bank_sync(uid)
            st.success(f"✅ Synced {n} transactions from bank")
            st.cache_data.clear()
            st.rerun()
        if st.button("✨ Generate 90-Day Demo Data", use_container_width=True):
            n = seed_90_days(uid)
            st.success(f"✅ Generated {n} transactions (Feb–Apr 2026)")
            st.cache_data.clear()
            st.rerun()

    st.subheader("💸 Monthly Budget")
    st.session_state.monthly_budget = st.number_input(
        "Budget (₹)", value=float(st.session_state.monthly_budget),
        min_value=1000.0, step=1000.0, key="budget_input"
    )

    st.markdown("---")
    st.subheader("🎯 Savings Goal")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        new_goal = st.text_input("Goal Name", value=st.session_state.goal_name, key="gname")
    with col_g2:
        new_goal_amt = st.number_input("Target (₹)", value=float(st.session_state.goal_amount),
                                        min_value=1000.0, step=5000.0, key="gamount")
    if new_goal != st.session_state.goal_name or new_goal_amt != st.session_state.goal_amount:
        st.session_state.goal_name = new_goal
        st.session_state.goal_amount = new_goal_amt
        save_prefs(uid, new_goal, new_goal_amt, st.session_state.food_limit)

    st.markdown("---")
    st.subheader("🚫 Category Limits")
    new_food = st.number_input("Monthly Food Limit (₹)", value=float(st.session_state.food_limit),
                                min_value=500.0, step=500.0, key="foodlim")
    if new_food != st.session_state.food_limit:
        st.session_state.food_limit = new_food
        save_prefs(uid, st.session_state.goal_name, st.session_state.goal_amount, new_food)
        st.success("✅ Food limit updated")

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for k in ['user','uid','prefs_loaded']:
            st.session_state[k] = None
        st.rerun()

# ─── MAIN DATA ─────────────────────────────────────────────────────────────────
df = fetch_txns(uid)

# ─── TABS (MOBILE STACK) ────────────────────────────────────────────────────────
tab_overview, tab_ai, tab_history, tab_scan = st.tabs([
    "📊 Overview", "🧠 AI Analysis", "📝 History", "📷 Scan"
])

# ─── OVERVIEW TAB ──────────────────────────────────────────────────────────────
with tab_overview:
    st.header("📊 Financial Overview")
    if df.empty:
        st.info("📭 No transactions. Use sidebar to generate demo data or scan receipts.")
    else:
        inc_tot = df[df['type']=='Income']['amount'].sum()
        exp_tot = df[df['type']=='Expense']['amount'].sum()
        net = inc_tot - exp_tot

        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Net Balance", fmt(net))
        m2.metric("📥 Total Income", fmt(inc_tot))
        m3.metric("📤 Total Expense", fmt(exp_tot))

        st.markdown("---")
        st.subheader("📈 Category Breakdown")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            exp_df = df[df['type']=='Expense']
            if not exp_df.empty:
                fig_exp = px.pie(exp_df, values='amount', names='category',
                                  title='Expense Breakdown', hole=0.4,
                                  color_discrete_sequence=px.colors.sequential.Reds)
                fig_exp.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_exp, use_container_width=True)
        with col_p2:
            inc_df = df[df['type']=='Income']
            if not inc_df.empty:
                fig_inc = px.pie(inc_df, values='amount', names='category',
                                  title='Income Sources', hole=0.4,
                                  color_discrete_sequence=px.colors.sequential.Greens)
                fig_inc.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_inc, use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Goal Progress")
        if st.session_state.goal_name:
            current_month_saving = inc_tot - exp_tot
            goal_pct = min((current_month_saving / st.session_state.goal_amount) * 100, 100) if st.session_state.goal_amount > 0 else 0
            cg1, cg2 = st.columns([3,1])
            with cg1:
                st.progress(goal_pct / 100)
            with cg2:
                st.metric("Saved", fmt(current_month_saving))
            st.info(f"📊 {goal_pct:.1f}% towards '{st.session_state.goal_name}' ({fmt(st.session_state.goal_amount)})")
        else:
            st.info("Set a Savings Goal in the sidebar to track progress.")

        anomalies = detect_anomalies(df)
        if not anomalies.empty:
            st.error(f"🚨 AI detected {len(anomalies)} suspicious transaction(s)")
        else:
            st.success("✅ All transactions verified by AI")

# ─── AI ANALYSIS TAB ────────────────────────────────────────────────────────────
with tab_ai:
    st.header("🧠 AI Analysis & Forecasting")
    if df.empty or len(df) < 3:
        st.info("📊 Add 3+ transactions to unlock AI forecasting")
    else:
        acc = train_cat_model(df)
        inc_f, exp_f = forecast_next_30(df)
        exp_df = df[df['type']=='Expense']
        top_cat = exp_df.groupby('category')['amount'].sum().idxmax() if not exp_df.empty else "N/A"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📈 Net Balance", fmt((df[df['type']=='Income']['amount'].sum() - exp_tot)))
        k2.metric("📅 Income Forecast", fmt(inc_f))
        k3.metric("📅 Expense Forecast", fmt(exp_f))
        k4.metric("🏷️ Top Category", top_cat)

        if acc:
            st.caption(f"🤖 Category model accuracy: {acc:.1%}")

        st.markdown("---")
        st.subheader("📉 Income vs Expense: Historical Trend + May 2026 Forecast")

        df_ts = df.copy()
        df_ts['date'] = pd.to_datetime(df_ts['date'])
        monthly_data = []
        for ttype in ['Income', 'Expense']:
            sub = df_ts[df_ts['type']==ttype]
            if not sub.empty:
                grp = sub.groupby(sub['date'].dt.to_period('M'))['amount'].sum().reset_index()
                for _, row in grp.iterrows():
                    monthly_data.append({'Month': str(row['date']), 'Type': ttype, 'Amount': row['amount']})

        if monthly_data:
            monthly_df = pd.DataFrame(monthly_data)
            fig = go.Figure()
            for ttype, col in [('Income','#00d4ff'), ('Expense','#ff4b4b')]:
                sub = monthly_df[monthly_df['Type']==ttype]
                fig.add_trace(go.Scatter(x=sub['Month'], y=sub['Amount'], mode='lines+markers',
                                         name=f'{ttype} Historical', line=dict(color=col, width=3)))
            forecast_month = "May 2026"
            fig.add_trace(go.Scatter(x=[forecast_month], y=[inc_f], mode='markers',
                                     name='Income Forecast', marker=dict(color='#00d4ff', size=12, symbol='star')))
            fig.add_trace(go.Scatter(x=[forecast_month], y=[exp_f], mode='markers',
                                     name='Expense Forecast', marker=dict(color='#ff4b4b', size=12, symbol='star')))
            fig.update_layout(
                title="Historical Trend with May 2026 Projection",
                xaxis_title="Month", yaxis_title=f"Amount ({CURRENCY})",
                template="plotly_dark", hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 AI Advisory (50/30/20 Rule)")
        tips = gen_advisory(df, st.session_state.monthly_budget,
                            st.session_state.goal_name, st.session_state.goal_amount,
                            inc_f, exp_f)
        for tip in tips:
            if "⚠️" in tip or "🚨" in tip:
                st.error(tip)
            else:
                st.success(tip) if "✅" in tip else st.info(tip)

# ─── HISTORY TAB ───────────────────────────────────────────────────────────────
with tab_history:
    st.header("📝 Transaction History")

    with st.form("add_txn", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            date = st.date_input("Date", datetime.now())
            amount = st.number_input("Amount (₹)", min_value=0.01, format="%.2f")
            ttype = st.selectbox("Type", ["Expense", "Income"])
        with c2:
            desc = st.text_input("Description")
            suggested, conf = predict_category(desc) if desc else ("Uncategorized", 0.0)
            category_opts = INCOME_CATS if ttype == "Income" else EXPENSE_CATS
            cat_idx = category_opts.index(suggested) if suggested in category_opts else 0
            category = st.selectbox("Category", category_opts, index=cat_idx)

        if suggested != "Uncategorized" and desc:
            if conf >= 0.6:
                st.info(f"🤖 AI suggests: **{suggested}** ({conf:.0%})")
            else:
                st.warning(f"⚠️ Low confidence: {conf:.0%}")

        submitted = st.form_submit_button("➕ Add Transaction", use_container_width=True)
        if submitted:
            if amount <= 0 or not desc:
                st.error("Amount & description required")
            else:
                save_txn(date, amount, category, ttype, desc, uid)
                train_cat_model(df)
                st.success("✅ Added!")
                st.cache_data.clear()
                st.rerun()

    st.markdown("---")
    st.subheader("🔍 Search & Filter")
    col_s1, col_s2 = st.columns([3,1])
    with col_s1:
        search = st.text_input("Search description", placeholder="Type keywords...")
    with col_s2:
        filter_type = st.selectbox("Type", ["All", "Income", "Expense"])

    if df.empty:
        st.info("No transactions yet")
    else:
        df_disp = df.copy()
        df_disp['date'] = pd.to_datetime(df_disp['date']).dt.date

        if search:
            df_disp = df_disp[df_disp['description'].str.contains(search, case=False, na=False)]
        if filter_type != "All":
            df_disp = df_disp[df_disp['type'] == filter_type]

        anomalies = detect_anomalies(df_disp)
        df_disp['AI Status'] = df_disp['id'].isin(anomalies['id']).map({True: '🚨 Anomaly', False: '✅ Verified'})

        df_disp['amount'] = df_disp['amount'].apply(fmt)

        display_cols = ['date', 'description', 'category', 'type', 'amount', 'AI Status']
        st.dataframe(df_disp[display_cols], use_container_width=True, height=400)

        st.markdown("---")
        st.subheader("🗑️ Delete Transaction")
        col_d1, col_d2 = st.columns([3,1])
        with col_d1:
            del_id = st.text_input("Document ID", key="del_id")
        with col_d2:
            st.write("")
            if st.button("Delete", key="del_btn"):
                if del_id:
                    if delete_txn(del_id, uid):
                        st.success(f"✅ Deleted {del_id}")
                        st.cache_data.clear()
                        st.rerun()
                else:
                    st.error("Enter a document ID")

# ─── SCAN TAB ──────────────────────────────────────────────────────────────────
with tab_scan:
    st.header("📷 Receipt Scanner (OCR)")
    st.info("📸 Point your camera at a receipt OR upload a photo. AI extracts amount & vendor automatically.")
    
    # Single column layout – maximum horizontal space for camera
    image = st.camera_input(
        "📸 Capture Receipt",
        label_visibility="collapsed"
    )
    
    # File uploader as alternative (wrapped in expander)
    with st.expander("📁 Upload from Gallery/Camera App", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose receipt image",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            # Convert uploaded file to PIL Image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Receipt", use_column_width=True)
    
    # Process image (either from camera or uploader)
    if image:
        # Ensure we have a PIL Image object
        if not isinstance(image, Image.Image):
            img = Image.open(image)
        else:
            img = image
        
        # Perform OCR
        ocr_result = perform_ocr(img)
        st.session_state.ocr_result = ocr_result
        
        # Show captured/preview image if from camera (upload already shown above)
        if 'uploaded_file' not in st.session_state or st.session_state.get('uploaded_file') is None:
            st.image(img, use_column_width=True, caption="Captured Receipt")
    
    st.markdown("---")
    st.subheader("📋 Extraction Result")
    
    if 'ocr_result' in st.session_state:
        res = st.session_state.ocr_result
        
        # Display results in two metrics side-by-side
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Vendor", res.get('vendor', 'Unknown'))
        with res_col2:
            amount_val = res.get('amount')
            st.metric("Amount", fmt(amount_val) if amount_val else "Not detected")
        
        # Full text expander
        with st.expander("📄 View Full OCR Text"):
            st.code(res.get('text', ''), language=None)
    else:
        st.info("👆 Take a photo or upload a file to see extracted data")
    
    # Save section
    if 'ocr_result' in st.session_state:
        st.markdown("---")
        st.subheader("✅ Confirm & Save Transaction")
        
        ocr = st.session_state.ocr_result
        amount_val = ocr.get('amount') or 0.0
        vendor_name = ocr.get('vendor') or 'Unknown'
        category_guess = ocr.get('category') or 'Other'
        
        # Centered confirmation form
        save_col1, save_col2, save_col3 = st.columns([1, 2, 1])
        with save_col2:
            save_date = st.date_input("Date", datetime.now(), key="scan_date")
            save_amount = st.number_input("Amount (₹)", value=float(amount_val), format="%.2f", key="scan_amount")
            save_desc = st.text_input("Description", value=f"Receipt - {vendor_name}", key="scan_desc")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                ttype = st.selectbox("Type", ["Expense", "Income"], key="scan_type")
            with col_t2:
                cat_opts = INCOME_CATS if ttype == "Income" else EXPENSE_CATS
                save_category = st.selectbox(
                    "Category", 
                    cat_opts, 
                    index=cat_opts.index(category_guess) if category_guess in cat_opts else 0,
                    key="scan_cat"
                )
            
            if st.button("💾 Save to Cloud", use_container_width=True):
                if save_amount > 0 and save_desc:
                    save_txn(save_date, save_amount, save_category, ttype, save_desc, uid)
                    st.success("✅ Transaction saved!")
                    # Clear OCR cache
                    if 'ocr_result' in st.session_state:
                        del st.session_state['ocr_result']
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Amount & description required")
