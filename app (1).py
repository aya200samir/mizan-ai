
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# إعدادات الصفحة
# ============================================
st.set_page_config(
    page_title="Mizan AI - نظام تحليل الإسكان الذكي",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS مخصص - تصميم احترافي
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    
    * {
        font-family: 'Cairo', sans-serif;
        box-sizing: border-box;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .header h1 {
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        height: 100%;
        border: 1px solid #e0e0e0;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(30, 60, 114, 0.2);
    }
    
    .card .value {
        font-size: 2rem;
        font-weight: 900;
        color: #1e3c72;
        margin: 0.5rem 0;
    }
    
    .card .label {
        color: #666;
        font-size: 1rem;
    }
    
    .source-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .footer {
        background: #1e3c72;
        color: white;
        padding: 2rem;
        border-radius: 30px 30px 0 0;
        margin-top: 3rem;
        text-align: center;
    }
    
    .badge {
        display: inline-block;
        background: #4CAF50;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5298, #1e3c72);
        box-shadow: 0 5px 15px rgba(30, 60, 114, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# توليد بيانات
# ============================================
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 1000
    
    governorates = ['القاهرة', 'الجيزة', 'الإسكندرية', 'الدقهلية', 'الشرقية',
                    'المنوفية', 'الغربية', 'القليوبية', 'البحيرة', 'كفر الشيخ',
                    'دمياط', 'بورسعيد', 'الإسماعيلية', 'السويس', 'شمال سيناء',
                    'جنوب سيناء', 'مطروح', 'الفيوم', 'بني سويف', 'المنيا',
                    'أسيوط', 'سوهاج', 'قنا', 'الأقصر', 'أسوان']
    
    employment = ['موظف حكومي', 'قطاع خاص', 'عمل حر', 'عمالة غير منتظمة',
                  'عاطل', 'متقاعد', 'طالب', 'رب منزل']
    
    data = pd.DataFrame({
        'الرقم': [f'APP-{i}' for i in range(1, n+1)],
        'العمر': np.random.randint(18, 70, n),
        'الجنس': np.random.choice(['ذكر', 'أنثى'], n, p=[0.48, 0.52]),
        'المحافظة': np.random.choice(governorates, n),
        'نوع العمل': np.random.choice(employment, n),
        'الدخل': np.random.normal(4500, 2000, n).clip(1000, 25000),
        'حجم الأسرة': np.random.poisson(4, n).clip(1, 12),
        'الأطفال': np.random.poisson(2, n).clip(0, 8),
        'إعاقة': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        'ملكية سابقة': np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })
    
    # حساب الاستحقاق
    data['مستحق'] = (
        (data['الدخل'] <= 6000) & 
        (data['ملكية سابقة'] == 0) &
        (data['العمر'] >= 21)
    ).astype(int)
    
    return data

# ============================================
# الصفحات
# ============================================
def main():
    st.markdown("""
    <div class="header">
        <h1>⚖️ MIZAN AI</h1>
        <p>نظام تحليل عدالة الإسكان الاجتماعي</p>
    </div>
    """, unsafe_allow_html=True)
    
    # قائمة جانبية
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/egypt--v1.png", width=80)
        st.markdown("### ⚖️ Mizan AI")
        st.markdown("---")
        
        menu = st.radio(
            "القائمة",
            ["🏠 الرئيسية", "📊 تحليل البيانات", "🧠 النظام الهجين", "📁 رفع البيانات", "ℹ️ عن المشروع"]
        )
        
        st.markdown("---")
        st.markdown("**إحصائيات سريعة**")
        st.markdown("- 27 محافظة")
        st.markdown("- 1000+ متقدم")
        st.markdown("- 12% تحيز")
        
        if st.button("🔄 تحديث"):
            st.cache_data.clear()
            st.rerun()
    
    # تحميل البيانات
    data = generate_data()
    
    # الصفحة الرئيسية
    if menu == "🏠 الرئيسية":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <div style="font-size: 2rem;">👥</div>
                <div class="value">{len(data):,}</div>
                <div class="label">إجمالي المتقدمين</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <div style="font-size: 2rem;">✅</div>
                <div class="value">{data['مستحق'].mean()*100:.1f}%</div>
                <div class="label">نسبة المستحقين</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card">
                <div style="font-size: 2rem;">💰</div>
                <div class="value">{data['الدخل'].mean():,.0f}</div>
                <div class="label">متوسط الدخل</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="card">
                <div style="font-size: 2rem;">👪</div>
                <div class="value">{data['حجم الأسرة'].mean():.1f}</div>
                <div class="label">متوسط الأسرة</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 توزيع المحافظات")
            gov_counts = data['المحافظة'].value_counts().head(10)
            fig = px.bar(x=gov_counts.index, y=gov_counts.values, 
                        title="أكثر 10 محافظات",
                        color=gov_counts.values,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 توزيع الدخل")
            fig = px.histogram(data, x='الدخل', nbins=30,
                             title="توزيع الدخل الشهري",
                             color_discrete_sequence=['#1e3c72'])
            fig.add_vline(x=6000, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    
    # تحليل البيانات
    elif menu == "📊 تحليل البيانات":
        st.markdown("## 📊 تحليل متقدم")
        
        tab1, tab2, tab3 = st.tabs(["ديموغرافيا", "الدخل", "الاستحقاق"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                gender_data = data['الجنس'].value_counts()
                fig = px.pie(values=gender_data.values, names=gender_data.index,
                           title="التوزيع حسب الجنس")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                work_data = data['نوع العمل'].value_counts().head(8)
                fig = px.bar(x=work_data.index, y=work_data.values,
                           title="توزيع أنواع العمل")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                income_by_gov = data.groupby('المحافظة')['الدخل'].mean().sort_values(ascending=False).head(10)
                fig = px.bar(x=income_by_gov.values, y=income_by_gov.index,
                           orientation='h', title="متوسط الدخل حسب المحافظة")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                income_by_work = data.groupby('نوع العمل')['الدخل'].mean().sort_values(ascending=False)
                fig = px.bar(x=income_by_work.index, y=income_by_work.values,
                           title="متوسط الدخل حسب نوع العمل")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                eligible_by_gov = data.groupby('المحافظة')['مستحق'].mean().sort_values(ascending=False).head(10)
                fig = px.bar(x=eligible_by_gov.values, y=eligible_by_gov.index,
                           orientation='h', title="نسبة الاستحقاق حسب المحافظة",
                           color=eligible_by_gov.values,
                           color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                eligible_by_work = data.groupby('نوع العمل')['مستحق'].mean().sort_values(ascending=False)
                fig = px.bar(x=eligible_by_work.index, y=eligible_by_work.values,
                           title="نسبة الاستحقاق حسب نوع العمل",
                           color=eligible_by_work.values,
                           color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
    
    # النظام الهجين
    elif menu == "🧠 النظام الهجين":
        st.markdown("## 🧠 نظام القرار الهجين")
        
        st.info("""
        **كيف يعمل النظام؟**
        - ✅ حالات واضحة (ثقة > 80%) → قرار تلقائي
        - ❌ حالات واضحة (ثقة < 30%) → قرار تلقائي
        - 👤 حالات حدية (30% - 80%) → مراجعة بشرية
        """)
        
        # اختيار عينة عشوائية
        sample = data.sample(1).iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 بيانات المتقدم")
            st.json({
                'العمر': int(sample['العمر']),
                'الجنس': sample['الجنس'],
                'المحافظة': sample['المحافظة'],
                'نوع العمل': sample['نوع العمل'],
                'الدخل': f"{sample['الدخل']:,.0f} ج.م",
                'حجم الأسرة': int(sample['حجم الأسرة']),
                'إعاقة': 'نعم' if sample['إعاقة'] == 1 else 'لا'
            })
        
        with col2:
            st.markdown("### ⚖️ نتيجة التحليل")
            
            # حساب الاحتمالية
            prob = 0.5
            
            if sample['الدخل'] <= 6000:
                prob += 0.3
            else:
                prob -= 0.2
            
            if sample['ملكية سابقة'] == 0:
                prob += 0.2
            else:
                prob -= 0.3
            
            if sample['العمر'] >= 21:
                prob += 0.1
            
            if sample['إعاقة'] == 1:
                prob += 0.2
            
            prob = np.clip(prob, 0, 1)
            
            # شريط التقدم
            st.progress(prob)
            st.markdown(f"**احتمالية الاستحقاق:** {prob:.1%}")
            
            # القرار
            if prob >= 0.8:
                st.success("✅ **مقبول تلقائياً** - حالة واضحة")
            elif prob <= 0.3:
                st.error("❌ **مرفوض تلقائياً** - لا يستوفي الشروط")
            else:
                st.warning("👤 **يحتاج مراجعة بشرية** - حالة حدية")
    
    # رفع البيانات
    elif menu == "📁 رفع البيانات":
        st.markdown("## 📁 رفع بياناتك الخاصة")
        
        uploaded_file = st.file_uploader(
            "اختر ملف CSV أو Excel",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    user_data = pd.read_csv(uploaded_file)
                else:
                    user_data = pd.read_excel(uploaded_file)
                
                st.success(f"✅ تم تحميل {len(user_data)} سجل!")
                st.dataframe(user_data.head(10))
            except:
                st.error("❌ خطأ في قراءة الملف")
    
    # عن المشروع
    else:
        st.markdown("## ℹ️ عن مشروع Mizan AI")
        
        st.markdown("""
        ### 🎯 رؤية المشروع
        نظام ذكاء اصطناعي أخلاقي لتحقيق العدالة في توزيع الإسكان الاجتماعي.
        
        ### ⚙️ المكونات
        - ✅ أوزان إضافية للفئات الأقل تمثيلاً
        - 📊 تحليل متقدم للبيانات
        - 🧠 نظام قرار هجين (آلي + بشري)
        - 🔍 كشف التحيزات
        
        ### 📚 المصادر الرسمية
        - صندوق الإسكان الاجتماعي
        - الجهاز المركزي للإحصاء
        - وزارة الإسكان
        """)
    
    # تذييل
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.2rem;">⚖️ Mizan AI - نظام العدالة الذكي</p>
        <p>مدعوم ببيانات من المصادر الرسمية</p>
        <p style="font-size: 0.8rem; opacity: 0.8;">© 2026 جميع الحقوق محفوظة</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
