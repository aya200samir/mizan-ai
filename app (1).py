import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================
# مكتبات التعلم الآلي (بدون xgboost)
# ============================================
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ============================================
# إعدادات الصفحة
# ============================================
st.set_page_config(
    page_title="Mizan AI - نظام التنبؤ الذكي",
    page_icon="🤖",
    layout="wide"
)

# ============================================
# CSS مخصص
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    
    * { font-family: 'Cairo', sans-serif; }
    
    .header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    
    .metric-box .value { font-size: 2rem; font-weight: 900; }
    .metric-box .label { font-size: 1rem; opacity: 0.9; }
    
    .footer {
        background: #1e3c72;
        color: white;
        padding: 1.5rem;
        border-radius: 30px 30px 0 0;
        margin-top: 3rem;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# توليد بيانات التدريب
# ============================================
@st.cache_data
def generate_data(n_samples=3000):
    """توليد بيانات تدريب"""
    np.random.seed(42)
    
    governorates = ['القاهرة', 'الجيزة', 'الإسكندرية', 'الدقهلية', 'الشرقية',
                    'أسيوط', 'سوهاج', 'قنا', 'الأقصر', 'أسوان']
    
    employment = ['موظف حكومي', 'قطاع خاص', 'عمل حر', 'عمالة غير منتظمة', 'عاطل']
    
    data = pd.DataFrame({
        'العمر': np.random.randint(18, 70, n_samples),
        'الجنس': np.random.choice(['ذكر', 'أنثى'], n_samples),
        'المحافظة': np.random.choice(governorates, n_samples),
        'نوع_العمل': np.random.choice(employment, n_samples),
        'الدخل': np.random.randint(2000, 15000, n_samples),
        'حجم_الأسرة': np.random.randint(1, 8, n_samples),
        'إعاقة': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'ملكية_سابقة': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    })
    
    # الهدف
    data['مستحق'] = (
        (data['الدخل'] < 6000) & 
        (data['ملكية_سابقة'] == 0) &
        (data['العمر'] >= 21)
    ).astype(int)
    
    # استثناءات
    special = (data['إعاقة'] == 1) & (data['الدخل'] <= 7000)
    data.loc[special, 'مستحق'] = 1
    
    return data

# ============================================
# تدريب النموذج
# ============================================
@st.cache_resource
def train_model(data):
    """تدريب Random Forest"""
    
    feature_cols = ['العمر', 'الدخل', 'حجم_الأسرة', 'إعاقة', 'ملكية_سابقة']
    
    # ترميز الفئويات
    data_encoded = data.copy()
    encoders = {}
    
    for col in ['الجنس', 'المحافظة', 'نوع_العمل']:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        encoders[col] = le
        feature_cols.append(col)
    
    X = data_encoded[feature_cols]
    y = data_encoded['مستحق']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return {
        'model': model,
        'encoders': encoders,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

# ============================================
# التنبؤ
# ============================================
def predict(model_pack, user_data):
    input_df = pd.DataFrame([user_data])
    
    for col, encoder in model_pack['encoders'].items():
        input_df[col] = encoder.transform(input_df[col])
    
    X_input = input_df[model_pack['feature_cols']]
    prob = model_pack['model'].predict_proba(X_input)[0][1]
    pred = model_pack['model'].predict(X_input)[0]
    
    return pred, prob

# ============================================
# الصفحة الرئيسية
# ============================================
def main():
    st.markdown("""
    <div class="header">
        <h1>⚖️ MIZAN AI</h1>
        <p>نظام التنبؤ بالاستحقاق - الإسكان الاجتماعي</p>
    </div>
    """, unsafe_allow_html=True)
    
    # تدريب النموذج
    with st.spinner("📊 جاري تجهيز النموذج..."):
        data = generate_data(3000)
        model_pack = train_model(data)
    
    # القائمة الجانبية
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/egypt--v1.png", width=80)
        menu = st.radio("", ["🏠 الرئيسية", "🧠 تنبؤ", "📊 تقييم"])
    
    if menu == "🏠 الرئيسية":
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("حالات التدريب", f"{len(data):,}")
        with col2: st.metric("دقة النموذج", f"{model_pack['metrics']['accuracy']*100:.1f}%")
        with col3: st.metric("F1 Score", f"{model_pack['metrics']['f1']*100:.1f}%")
        with col4: st.metric("نسبة المستحقين", f"{data['مستحق'].mean()*100:.1f}%")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=data['مستحق'].value_counts().values,
                        names=['غير مستحق', 'مستحق'])
            st.plotly_chart(fig)
        with col2:
            fig = px.box(data, x='مستحق', y='الدخل', color='مستحق')
            st.plotly_chart(fig)
    
    elif menu == "🧠 تنبؤ":
        st.markdown("## 🧠 تنبؤ الاستحقاق")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("العمر", 18, 70, 35)
            gender = st.selectbox("الجنس", ['ذكر', 'أنثى'])
            governorate = st.selectbox("المحافظة", 
                ['القاهرة', 'الجيزة', 'الإسكندرية', 'الدقهلية', 'الشرقية', 'أسيوط'])
            employment = st.selectbox("نوع العمل", 
                ['موظف حكومي', 'قطاع خاص', 'عمل حر', 'عمالة غير منتظمة', 'عاطل'])
        
        with col2:
            income = st.number_input("الدخل الشهري", 2000, 15000, 4500)
            family_size = st.number_input("حجم الأسرة", 1, 7, 4)
            disability = st.checkbox("لديه إعاقة")
            previous = st.checkbox("ملكية سابقة")
        
        if st.button("🔮 تنبؤ"):
            user_data = {
                'العمر': age, 'الجنس': gender, 'المحافظة': governorate,
                'نوع_العمل': employment, 'الدخل': income, 'حجم_الأسرة': family_size,
                'إعاقة': 1 if disability else 0, 'ملكية_سابقة': 1 if previous else 0
            }
            pred, prob = predict(model_pack, user_data)
            
            if pred == 1:
                st.success(f"### ✅ مستحق (ثقة: {prob*100:.1f}%)")
            else:
                st.error(f"### ❌ غير مستحق (ثقة: {(1-prob)*100:.1f}%)")
            st.progress(prob)
    
    elif menu == "📊 تقييم":
        st.markdown("## 📊 تقييم النموذج")
        st.dataframe(pd.DataFrame([
            {'المقياس': 'الدقة', 'القيمة': f"{model_pack['metrics']['accuracy']*100:.2f}%"},
            {'المقياس': 'Precision', 'القيمة': f"{model_pack['metrics']['precision']*100:.2f}%"},
            {'المقياس': 'Recall', 'القيمة': f"{model_pack['metrics']['recall']*100:.2f}%"},
            {'المقياس': 'F1 Score', 'القيمة': f"{model_pack['metrics']['f1']*100:.2f}%"}
        ]))
        
        cm = confusion_matrix(model_pack['y_test'], model_pack['y_pred'])
        fig = px.imshow(cm, text_auto=True, x=['غير مستحق', 'مستحق'], y=['غير مستحق', 'مستحق'])
        st.plotly_chart(fig)
    
    st.markdown("""
    <div class="footer">
        <p>⚖️ Mizan AI - نظام التنبؤ بالاستحقاق | © 2026</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
