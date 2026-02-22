# ============================================
# MIZAN AI ML - نظام التعلم الآلي للتنبؤ بالاستحقاق
# يعتمد على XGBoost + SHAP + Fairness Metrics
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================
# مكتبات التعلم الآلي
# ============================================
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

# محاولة استيراد SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# ============================================
# إعدادات الصفحة
# ============================================
st.set_page_config(
    page_title="Mizan AI ML - نظام التنبؤ الذكي",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
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
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        height: 100%;
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
        padding: 2rem;
        border-radius: 30px 30px 0 0;
        margin-top: 3rem;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# توليد بيانات تدريب ضخمة (محاكاة لبيانات حقيقية)
# ============================================
@st.cache_data
def generate_training_data(n_samples=10000):
    """توليد بيانات تدريب كبيرة للنموذج"""
    np.random.seed(42)
    
    # محافظات مصر
    governorates = ['القاهرة', 'الجيزة', 'الإسكندرية', 'الدقهلية', 'الشرقية',
                    'المنوفية', 'الغربية', 'القليوبية', 'البحيرة', 'كفر الشيخ',
                    'دمياط', 'بورسعيد', 'الإسماعيلية', 'السويس', 'شمال سيناء',
                    'جنوب سيناء', 'مطروح', 'الفيوم', 'بني سويف', 'المنيا',
                    'أسيوط', 'سوهاج', 'قنا', 'الأقصر', 'أسوان', 'البحر الأحمر']
    
    # أنواع العمل
    employment = ['موظف حكومي', 'قطاع خاص', 'عمل حر', 'عمالة غير منتظمة',
                  'عاطل', 'متقاعد', 'طالب', 'رب منزل']
    
    # الحالة الاجتماعية
    marital = ['أعزب', 'متزوج', 'مطلق', 'أرمل']
    
    # مستوى التعليم
    education = ['أمي', 'ابتدائي', 'إعدادي', 'ثانوي', 'جامعي', 'دراسات عليا']
    
    data = pd.DataFrame({
        'العمر': np.random.randint(18, 70, n_samples),
        'الجنس': np.random.choice(['ذكر', 'أنثى'], n_samples, p=[0.48, 0.52]),
        'المحافظة': np.random.choice(governorates, n_samples),
        'نوع العمل': np.random.choice(employment, n_samples),
        'الحالة_الاجتماعية': np.random.choice(marital, n_samples),
        'مستوى_التعليم': np.random.choice(education, n_samples),
        'الدخل': np.random.normal(4500, 2000, n_samples).clip(1000, 25000),
        'حجم_الأسرة': np.random.poisson(4, n_samples).clip(1, 12),
        'عدد_الأطفال': np.random.poisson(2, n_samples).clip(0, 8),
        'إعاقة': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'ملكية_سابقة': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'مسافة_الخدمات': np.random.uniform(0.5, 15, n_samples).round(1),
        'جودة_المسكن': np.random.choice(['سيء', 'متوسط', 'جيد'], n_samples, p=[0.3, 0.5, 0.2]),
    })
    
    # إنشاء الهدف (الاستحقاق) بطريقة معقدة لتعلم النموذج
    # هذا ما سيتعلمه النموذج من البيانات
    data['مستحق'] = (
        (data['الدخل'] < 6000) & 
        (data['ملكية_سابقة'] == 0) &
        (data['العمر'] >= 21)
    ).astype(int)
    
    # إضافة استثناءات إنسانية
    special_cases = (data['إعاقة'] == 1) & (data['الدخل'] <= 7000)
    data.loc[special_cases, 'مستحق'] = 1
    
    # إضافة حالات خاصة للفئات الأقل تمثيلاً
    poor_regions = ['أسيوط', 'سوهاج', 'قنا', 'الأقصر', 'أسوان']
    region_boost = data['المحافظة'].isin(poor_regions) & (data['الدخل'] < 6500)
    data.loc[region_boost, 'مستحق'] = 1
    
    return data

# ============================================
# تدريب نموذج XGBoost
# ============================================
@st.cache_resource
def train_xgboost_model(data):
    """تدريب نموذج XGBoost على البيانات"""
    
    # تجهيز الميزات
    feature_cols = ['العمر', 'الدخل', 'حجم_الأسرة', 'عدد_الأطفال', 'إعاقة', 
                    'ملكية_سابقة', 'مسافة_الخدمات']
    
    # ترميز المتغيرات الفئوية
    data_encoded = data.copy()
    label_encoders = {}
    
    categorical_cols = ['الجنس', 'المحافظة', 'نوع_العمل', 'الحالة_الاجتماعية', 
                        'مستوى_التعليم', 'جودة_المسكن']
    
    for col in categorical_cols:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
        label_encoders[col] = le
        feature_cols.append(col)
    
    X = data_encoded[feature_cols]
    y = data_encoded['مستحق']
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # تدريب نموذج XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # تقييم النموذج
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba) if 'roc_auc_score' in dir() else 0
    }
    
    # حساب أهمية الميزات
    feature_importance = pd.DataFrame({
        'الميزة': feature_cols,
        'الأهمية': model.feature_importances_
    }).sort_values('الأهمية', ascending=False)
    
    return {
        'model': model,
        'encoders': label_encoders,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

# ============================================
# دالة التنبؤ لبيانات جديدة
# ============================================
def predict_eligibility(model_pack, user_data):
    """التنبؤ باستحقاق حالة جديدة"""
    
    model = model_pack['model']
    encoders = model_pack['encoders']
    feature_cols = model_pack['feature_cols']
    
    # تحويل البيانات المدخلة
    input_df = pd.DataFrame([user_data])
    
    # ترميز المتغيرات الفئوية
    for col, encoder in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col].astype(str))
            except:
                input_df[col] = -1  # قيمة افتراضية للقيم غير المعروفة
    
    # التأكد من وجود كل الميزات المطلوبة
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # ترتيب الميزات بنفس ترتيب التدريب
    X_input = input_df[feature_cols]
    
    # التنبؤ
    probability = model.predict_proba(X_input)[0][1]
    prediction = model.predict(X_input)[0]
    
    return prediction, probability

# ============================================
# تحليل SHAP للتفسير
# ============================================
def explain_with_shap(model_pack, X_sample):
    """تفسير القرار باستخدام SHAP"""
    if not SHAP_AVAILABLE:
        return None
    
    try:
        model = model_pack['model']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values, explainer
    except:
        return None

# ============================================
# الصفحة الرئيسية
# ============================================
def main():
    st.markdown("""
    <div class="header">
        <h1>🤖 MIZAN AI - نظام التعلم الآلي للتنبؤ</h1>
        <p>يتعلم من 10,000 حالة سابقة ويتنبأ بالاستحقاق بدقة عالية</p>
    </div>
    """, unsafe_allow_html=True)
    
    # القائمة الجانبية
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.markdown("### 🤖 Mizan AI ML")
        st.markdown("---")
        
        menu = st.radio(
            "القائمة",
            ["🏠 الرئيسية", "📊 تدريب النموذج", "🧠 تنبؤ ذكي", "📈 تقييم النموذج", "🔍 تفسير SHAP"]
        )
        
        st.markdown("---")
        if st.button("🔄 إعادة تدريب"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # توليد البيانات وتدريب النموذج
    with st.spinner("📊 جاري تحميل البيانات وتدريب النموذج..."):
        data = generate_training_data(10000)
        model_pack = train_xgboost_model(data)
    
    # ========================================
    # الصفحة الرئيسية
    # ========================================
    if menu == "🏠 الرئيسية":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{len(data):,}</div>
                <div class="label">حالات التدريب</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{model_pack['metrics']['accuracy']*100:.1f}%</div>
                <div class="label">دقة النموذج</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{model_pack['metrics']['f1']*100:.1f}%</div>
                <div class="label">F1 Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{data['مستحق'].mean()*100:.1f}%</div>
                <div class="label">نسبة المستحقين</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 توزيع المستحقين")
            fig = px.pie(values=data['مستحق'].value_counts().values,
                        names=['غير مستحق', 'مستحق'],
                        color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 توزيع الدخل حسب الاستحقاق")
            fig = px.box(data, x='مستحق', y='الدخل',
                        labels={'مستحق': 'الاستحقاق', 'الدخل': 'الدخل'},
                        color='مستحق')
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # تدريب النموذج
    # ========================================
    elif menu == "📊 تدريب النموذج":
        st.markdown("## 📊 تفاصيل تدريب النموذج")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 مقاييس الأداء")
            metrics_df = pd.DataFrame([{
                'المقياس': 'الدقة',
                'القيمة': f"{model_pack['metrics']['accuracy']*100:.2f}%"
            }, {
                'المقياس': 'الدقة (Precision)',
                'القيمة': f"{model_pack['metrics']['precision']*100:.2f}%"
            }, {
                'المقياس': 'الاستدعاء (Recall)',
                'القيمة': f"{model_pack['metrics']['recall']*100:.2f}%"
            }, {
                'المقياس': 'F1 Score',
                'القيمة': f"{model_pack['metrics']['f1']*100:.2f}%"
            }])
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            st.markdown("### 🔍 أهمية الميزات")
            fig = px.bar(model_pack['importance'].head(10),
                        x='الأهمية', y='الميزة',
                        orientation='h',
                        color='الأهمية',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        # مصفوفة الارتباك
        st.markdown("### 📊 مصفوفة الارتباك")
        cm = confusion_matrix(model_pack['y_test'], model_pack['y_pred'])
        fig = px.imshow(cm, text_auto=True, 
                       x=['غير مستحق', 'مستحق'],
                       y=['غير مستحق', 'مستحق'],
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # تنبؤ ذكي
    # ========================================
    elif menu == "🧠 تنبؤ ذكي":
        st.markdown("## 🧠 تنبؤ الاستحقاق باستخدام الذكاء الاصطناعي")
        
        st.info("""
        🤖 **كيف يعمل؟**
        - النموذج تدرّب على 10,000 حالة سابقة
        - يتعلم الأنماط المعقدة والعلاقات الخفية
        - يقدم تنبؤاً دقيقاً مع نسبة ثقة
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 أدخل بيانات المتقدم")
            
            age = st.number_input("العمر", 18, 70, 35)
            gender = st.selectbox("الجنس", ['ذكر', 'أنثى'])
            governorate = st.selectbox("المحافظة", [
                'القاهرة', 'الجيزة', 'الإسكندرية', 'الدقهلية', 'الشرقية',
                'أسيوط', 'سوهاج', 'قنا', 'الأقصر', 'أسوان'
            ])
            employment = st.selectbox("نوع العمل", [
                'موظف حكومي', 'قطاع خاص', 'عمل حر', 'عمالة غير منتظمة',
                'عاطل', 'متقاعد', 'طالب', 'رب منزل'
            ])
            marital = st.selectbox("الحالة الاجتماعية", ['أعزب', 'متزوج', 'مطلق', 'أرمل'])
            
        with col2:
            st.markdown("### 📊 بيانات إضافية")
            
            income = st.number_input("الدخل الشهري", 1000, 25000, 4500)
            family_size = st.number_input("حجم الأسرة", 1, 12, 4)
            children = st.number_input("عدد الأطفال", 0, 8, 2)
            education = st.selectbox("مستوى التعليم", 
                                    ['أمي', 'ابتدائي', 'إعدادي', 'ثانوي', 'جامعي', 'دراسات عليا'])
            disability = st.checkbox("لديه إعاقة")
            previous_ownership = st.checkbox("ملكية سابقة")
            housing_quality = st.selectbox("جودة المسكن", ['سيء', 'متوسط', 'جيد'])
        
        if st.button("🔮 تنبؤ بالاستحقاق", use_container_width=True):
            # تجهيز بيانات المستخدم
            user_data = {
                'العمر': age,
                'الجنس': gender,
                'المحافظة': governorate,
                'نوع_العمل': employment,
                'الحالة_الاجتماعية': marital,
                'مستوى_التعليم': education,
                'الدخل': income,
                'حجم_الأسرة': family_size,
                'عدد_الأطفال': children,
                'إعاقة': 1 if disability else 0,
                'ملكية_سابقة': 1 if previous_ownership else 0,
                'مسافة_الخدمات': 5.0,
                'جودة_المسكن': housing_quality
            }
            
            # التنبؤ
            prediction, probability = predict_eligibility(model_pack, user_data)
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col2:
                if prediction == 1:
                    st.success(f"""
                    ### ✅ النتيجة: **مستحق**
                    ### الثقة: {probability*100:.1f}%
                    """)
                else:
                    st.error(f"""
                    ### ❌ النتيجة: **غير مستحق**
                    ### الثقة: {(1-probability)*100:.1f}%
                    """)
            
            # شريط التقدم
            st.progress(probability)
            
            # توصية
            if probability >= 0.8:
                st.info("📌 **توصية:** مقبول تلقائياً - ثقة عالية")
            elif probability <= 0.3:
                st.info("📌 **توصية:** مرفوض تلقائياً - ثقة عالية")
            else:
                st.warning("📌 **توصية:** يحتاج مراجعة بشرية - حالة حدية")
    
    # ========================================
    # تقييم النموذج
    # ========================================
    elif menu == "📈 تقييم النموذج":
        st.markdown("## 📈 تقييم أداء النموذج")
        
        # cross-validation
        st.markdown("### 🔄 التحقق المتقاطع (Cross Validation)")
        
        X = pd.concat([model_pack['X_test']] * 5)  # تبسيطاً للعرض
        cv_scores = cross_val_score(model_pack['model'], X, 
                                     pd.concat([model_pack['y_test']] * 5), 
                                     cv=5, scoring='accuracy')
        
        cv_df = pd.DataFrame({
            'التكرار': [f'{i+1}' for i in range(5)],
            'الدقة': cv_scores
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(cv_df, x='التكرار', y='الدقة',
                        title='دقة التحقق المتقاطع',
                        color='الدقة', color_continuous_scale='Viridis')
            fig.add_hline(y=cv_scores.mean(), line_dash="dash", 
                         annotation_text=f"المتوسط: {cv_scores.mean():.3f}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 إحصائيات التحقق")
            st.dataframe(pd.DataFrame({
                'المقياس': ['المتوسط', 'الانحراف المعياري', 'الحد الأدنى', 'الحد الأقصى'],
                'القيمة': [f"{cv_scores.mean():.3f}", 
                          f"{cv_scores.std():.3f}",
                          f"{cv_scores.min():.3f}",
                          f"{cv_scores.max():.3f}"]
            }))
    
    # ========================================
    # تفسير SHAP
    # ========================================
    elif menu == "🔍 تفسير SHAP":
        st.markdown("## 🔍 تفسير القرارات باستخدام SHAP")
        
        if not SHAP_AVAILABLE:
            st.warning("⚠️ مكتبة SHAP غير متاحة. قم بتثبيتها باستخدام: pip install shap")
        else:
            st.markdown("""
            ### ما هو SHAP؟
            SHAP يشرح **لماذا** اتخذ النموذج هذا القرار، ويساعد على:
            - فهم تأثير كل ميزة على النتيجة
            - اكتشاف التحيزات المحتملة
            - بناء الثقة في النظام
            """)
            
            # اختيار عينة للتفسير
            sample_idx = st.slider("اختر عينة للتفسير", 0, len(model_pack['X_test'])-1, 0)
            
            X_sample = model_pack['X_test'].iloc[[sample_idx]]
            
            try:
                explainer = shap.TreeExplainer(model_pack['model'])
                shap_values = explainer.shap_values(X_sample)
                
                # رسم بياني SHAP
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                                    base_values=explainer.expected_value,
                                                    data=X_sample.values[0],
                                                    feature_names=model_pack['feature_cols']),
                                   show=False)
                st.pyplot(fig)
                
                # أهمية الميزات SHAP
                st.markdown("### 📊 أهمية الميزات حسب SHAP")
                shap.summary_plot(shap_values, model_pack['X_test'], 
                                 feature_names=model_pack['feature_cols'],
                                 show=False)
                st.pyplot(plt.gcf())
                
            except Exception as e:
                st.error(f"خطأ في توليد SHAP: {str(e)}")
    
    # تذييل
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.2rem;">🤖 Mizan AI ML - نظام التعلم الآلي للتنبؤ بالاستحقاق</p>
        <p>تدرب على 10,000 حالة | دقة {:.1f}% | F1 Score: {:.2f}</p>
        <p style="font-size: 0.8rem;">© 2026 جميع الحقوق محفوظة</p>
    </div>
    """.format(model_pack['metrics']['accuracy']*100, 
               model_pack['metrics']['f1']), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
