import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="סימולטור מנהרת רוח AI", layout="wide")

st.title("🛩️ סימולטור מנהרת רוח ואופטימיזציה גנרטיבית")
st.write("מערכת AI למיטוב פרופיל כנף בזמן אמת")

# --- סרגל צד: הגדרות תנאי טיסה ---
st.sidebar.header("תנאי ניסוי")
mach = st.sidebar.slider("מהירות (Mach)", 0.1, 0.85, 0.5)
aoa = st.sidebar.slider("זווית התקפה (Degrees)", 0, 15, 4)
min_thickness = st.sidebar.slider("עובי מינימלי מותר (%)", 5, 20, 10)

run_button = st.sidebar.button("🚀 הרץ אופטימיזציה")

# --- פונקציות עזר ודמיון AI ---
def get_initial_airfoil():
    # יצירת פרופיל כנף בסיסי (NACA 0012)
    x = np.linspace(0, 1, 100)
    # נוסחת עובי פשוטה לצורך הדגמה
    yt = 5 * 0.12 * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    return x, yt

def predict_aerodynamics(x, y, mach, aoa):
    """
    כאן מתבצעת הקריאה למודל ה-AI שלכם (למשל PyTorch או NeuralFoil).
    כרגע הפונקציה מחזירה ערכים דמויים לצורך הדגמה.
    """
    max_thickness = np.max(y) * 2 * 100
    cl = 2 * np.pi * np.radians(aoa) + (mach * 0.2)
    cd = 0.008 + 0.05 * (cl**2) + (max_thickness * 0.0005)
    ld_ratio = cl / cd
    return cl, cd, ld_ratio

# --- אזור תצוגה ראשי ---
col1, col2 = st.columns([2, 1])

with col1:
    chart_placeholder = st.empty()

with col2:
    st.subheader("מדדי ביצועים")
    metric_ld = st.empty()
    metric_cl = st.empty()
    metric_cd = st.empty()

# מצב התחלתי
x, y_top = get_initial_airfoil()
y_bottom = -y_top

def render_plot(x, y_t, y_b, title="פרופיל כנף"):
    fig = go.Figure()
    # פרופיל עליון ותחתון
    fig.add_trace(go.Scatter(x=x, y=y_t, mode='lines', name='פרופיל עליון', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=y_b, mode='lines', name='פרופיל תחתון', line=dict(color='blue')))
    
    fig.update_layout(
        title=title,
        xaxis_title="X (אורך מיתר)",
        yaxis_title="Y (גובה)",
        yaxis=dict(scaleanchor="x", scaleratio=1), # שומר על יחס 1:1 בדיאגרמה
        height=400,
        showlegend=False
    )
    return fig

# הצגה ראשונית
cl, cd, ld = predict_aerodynamics(x, y_top, mach, aoa)
chart_placeholder.plotly_chart(render_plot(x, y_top, y_bottom, "מצב התחלתי"), use_container_width=True)
metric_ld.metric("יחס עילוי/גרר (L/D)", f"{ld:.2f}")
metric_cl.metric("מקדם עילוי (Cl)", f"{cl:.3f}")
metric_cd.metric("מקדם גרר (Cd)", f"{cd:.4f}")

# --- לולאת אופטימיזציה בלחיצה על הכפתור ---
if run_button:
    st.sidebar.text("מריץ אופטימיזציה...")
    
    # סימולציה של לולאת הנגזרות/אלגוריתם האופטימיזציה
    for step in range(1, 21):
        # עדכון הצורה בצעדים קטנים (בפועל: הנגזרות מגיעות מ-PyTorch)
        y_top = y_top * 0.98 + (0.02 * np.sin(np.pi * x) * 0.15)
        y_bottom = -y_top
        
        # חישוב מחדש ב-AI
        cl, cd, ld = predict_aerodynamics(x, y_top, mach, aoa)
        
        # עדכון השרטוט והמדדים בזמן אמת
        chart_placeholder.plotly_chart(render_plot(x, y_top, y_bottom, f"צעד אופטימיזציה: {step}/20"), use_container_width=True)
        metric_ld.metric("יחס עילוי/גרר (L/D)", f"{ld:.2f}", delta=f"{ld - 15:.2f}")
        metric_cl.metric("מקדם עילוי (Cl)", f"{cl:.3f}")
        metric_cd.metric("מקדם גרר (Cd)", f"{cd:.4f}")
        
        time.sleep(0.1) # השהייה קלה ליצירת האפקט הוויזואלי
        
    st.success("האופטימיזציה הושלמה בהצלחה!")
