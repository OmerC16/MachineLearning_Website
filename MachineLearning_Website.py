import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="מנהרת רוח אינטראקטיבית", layout="wide")

st.title("🌬️ סימולציית זרימת חלקיקי אוויר (Particle Streamlines)")

# --- סרגל צד ---
st.sidebar.header("פרמטרי זרימה")
aoa = st.sidebar.slider("זווית התקפה (Degrees)", -5, 20, 10)
speed = st.sidebar.slider("מהירות רוח", 1.0, 5.0, 3.0)
num_particles = st.sidebar.slider("כמות חלקיקי עשן", 50, 300, 150)

st.sidebar.write("---")
is_animating = st.sidebar.checkbox("הפעל אנימציה", value=True)

# --- הגדרת גיאומטרית הכנף ---
def get_airfoil_coords(aoa_deg):
    x = np.linspace(0, 1, 60)
    yt = 5 * 0.12 * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    
    rad = np.radians(-aoa_deg)
    # פרופיל עליון ותחתון
    x_top, y_top = x, yt
    x_bot, y_bot = x, -yt
    
    # סיבוב לפי זווית ההתקפה
    xt_rot = x_top * np.cos(rad) - y_top * np.sin(rad)
    yt_rot = x_top * np.sin(rad) + y_top * np.cos(rad)
    
    xb_rot = x_bot * np.cos(rad) - y_bot * np.sin(rad)
    yb_rot = x_bot * np.sin(rad) + y_bot * np.cos(rad)
    
    return xt_rot, yt_rot, xb_rot, yb_rot

# --- חישוב מודל שדה מהירויות (AI / Physics Backend) ---
def get_velocity_at_point(px, py, aoa_deg, base_speed):
    """
    פונקציה זו מחזירה את וקטור המהירות (vx, vy) בכל נקודה במרחב.
    כאן משלבים את ניבוי ה-AI (U-Net).
    """
    rad = np.radians(aoa_deg)
    vx = base_speed
    vy = 0.0
    
    # עקירת זרימה מעל הכנף (Lift Effect)
    if -0.2 < px < 1.2:
        dist_from_surface = np.abs(py - (px * np.sin(-rad)))
        if dist_from_surface < 0.4:
            # האצת זרימה מעל הכנף
            if py > 0:
                vx += base_speed * (0.4 + 0.03 * aoa_deg) / (dist_from_surface + 0.2)
                vy += base_speed * np.sin(-rad) * 0.5
            # הזדקרות וסחרור בזוויות גבוהות
            if aoa_deg > 12 and px > 0.4 and py > 0:
                vy += np.random.uniform(-1.0, 1.0) * base_speed * 0.5
                vx *= 0.5
                
    return vx, vy

# --- אתחול מיקומי החלקיקים ---
if "px" not in st.session_state or len(st.session_state.px) != num_particles:
    st.session_state.px = np.random.uniform(-0.8, 1.5, num_particles)
    st.session_state.py = np.random.uniform(-0.5, 0.5, num_particles)

plot_spot = st.empty()

# --- לולאת אנימציה ---
step = 0
while is_animating:
    step += 1
    
    # עדכון מיקום כל חלקיק לפי שדה המהירויות
    new_px = []
    new_py = []
    
    for i in range(num_particles):
        x_curr = st.session_state.px[i]
        y_curr = st.session_state.py[i]
        
        vx, vy = get_velocity_at_point(x_curr, y_curr, aoa, speed)
        
        dt = 0.02
        x_next = x_curr + vx * dt
        y_next = y_curr + vy * dt
        
        # אם חלקיק יצא מגבולות המסך - מחזירים אותו להתחלה (מימין/משמאל)
        if x_next > 1.5:
            x_next = -0.8
            y_next = np.random.uniform(-0.5, 0.5)
            
        new_px.append(x_next)
        new_py.append(y_next)
        
    st.session_state.px = np.array(new_px)
    st.session_state.py = np.array(new_py)
    
    # רנדור השרטוט
    xt_rot, yt_rot, xb_rot, yb_rot = get_airfoil_coords(aoa)
    
    fig = go.Figure()
    
    # 1. ציור החלקיקים (אוויר / עשן)
    fig.add_trace(go.Scatter(
        x=st.session_state.px,
        y=st.session_state.py,
        mode='markers',
        marker=dict(size=4, color='cyan', opacity=0.8),
        name='Air Flow'
    ))
    
    # 2. ציור הכנף
    fig.add_trace(go.Scatter(
        x=np.concatenate([xt_rot, xb_rot[::-1]]),
        y=np.concatenate([yt_rot, yb_rot[::-1]]),
        fill='toself',
        fillcolor='darkgray',
        line=dict(color='white', width=2),
        name='Airfoil'
    ))
    
    fig.update_layout(
        title=f"אנימציית זרימה בזמן אמת | זווית התקפה: {aoa}°",
        xaxis=dict(range=[-0.8, 1.5], showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.6, 0.6], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        height=500,
        showlegend=False
    )
    
    plot_spot.plotly_chart(fig, use_container_width=True)
    time.sleep(0.01) # קצב רענון
