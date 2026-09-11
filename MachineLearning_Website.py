import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="CFD Virtual Wind Tunnel", layout="wide")

st.title("🌬️ סימולטור CFD בזמן אמת - שדה זרימה ולחצים")

# --- פרמטרים בסרגל הצד ---
st.sidebar.header("הגדרות ניסוי")
aoa = st.sidebar.slider("זווית התקפה (זווית α)", -5, 20, 8)
velocity = st.sidebar.slider("מהירות אוויר חופשי (m/s)", 10, 100, 50)
visual_type = st.sidebar.radio("תצוגה ויזואלית", ["שדה לחצים (Pressure)", "שדה מהירות (Velocity Magnitude)"])

# --- יצירת רשת נקודות דו-ממדית (Grid) ---
nx, ny = 120, 60
x = np.linspace(-0.5, 1.5, nx)
y = np.linspace(-0.5, 0.5, ny)
X, Y = np.meshgrid(x, y)

# --- פונקציה לייצוג גיאומטרית כנף (NACA 0012) ---
def get_airfoil_mask(X, Y, aoa_deg):
    rad = np.radians(-aoa_deg)
    # סיבוב הקואורדינטות לפי זווית ההתקפה
    X_rot = X * np.cos(rad) - Y * np.sin(rad)
    Y_rot = X * np.sin(rad) + Y * np.cos(rad)
    
    mask = (X_rot >= 0) & (X_rot <= 1)
    x_c = np.where(mask, X_rot, 0)
    yt = 5 * 0.12 * (0.2969*np.sqrt(x_c) - 0.1260*x_c - 0.3516*x_c**2 + 0.2843*x_c**3 - 0.1015*x_c**4)
    
    inside = mask & (np.abs(Y_rot) <= yt)
    return inside, X_rot, Y_rot

# --- סימולציית שדה זרימה באמצעות מודל (מתורגם למשוואות פוטנציאליות / AI) ---
def compute_cfd_fields(X, Y, velocity, aoa_deg):
    """
    כאן משלבים את מודל ה-U-Net שלכם.
    במקדמים הפיזיקליים שלהלן מחושב קירוב של פוטנציאל זרימה לצורך הצגה ויזואלית.
    """
    rad = np.radians(aoa_deg)
    inside, X_rot, Y_rot = get_airfoil_mask(X, Y, aoa_deg)
    
    # חישוב קירוב מהירויות U_x ו-U_y
    r2 = X**2 + Y**2 + 0.01
    U_x = velocity * np.cos(rad) + (X / r2) * 0.05
    U_y = velocity * np.sin(rad) - (Y / r2) * 0.05
    
    # עקמומיות והפרעת זרימה מעל הכנף (אפקט עילוי)
    over_airfoil = (X > 0) & (X < 1) & (Y > 0)
    U_x[over_airfoil] += velocity * (0.15 + 0.02 * aoa_deg)
    
    # אזור הפרעות וסחרור במקרה של הזדקרות (זווית גבוהה)
    if aoa_deg > 12:
        wake_zone = (X > 0.5) & (Y > 0) & (Y < 0.3)
        U_x[wake_zone] *= 0.3 # ירידה חדה במהירות מאחור
        U_y[wake_zone] += np.random.normal(0, velocity * 0.1, np.sum(wake_zone)) # טורבולנציה
        
    Vel_mag = np.sqrt(U_x**2 + U_y**2)
    Vel_mag[inside] = 0 # מהירות אפס בתוך גוף הכנף
    
    # חישוב לחץ לפי חוק ברנולי: P = P0 - 0.5 * rho * V^2
    rho = 1.225
    Pressure = 0.5 * rho * (velocity**2 - Vel_mag**2)
    Pressure[inside] = np.nan # הסתרת ערכים בתוך המבנה
    
    return Vel_mag, Pressure, inside

# --- הרצת החישוב ---
Vel_mag, Pressure, inside = compute_cfd_fields(X, Y, velocity, aoa)

# --- בניית השרטוט ב-Plotly ---
field_to_show = Pressure if visual_type == "שדה לחצים (Pressure)" else Vel_mag
color_scale = "RdBu_r" if visual_type == "שדה לחצים (Pressure)" else "Turbo"

fig = go.Figure()

# 1. מפת חום מוחלקת ללחץ/מהירות (Contour Plot)
fig.add_trace(go.Contour(
    z=field_to_show,
    x=x,
    y=y,
    colorscale=color_scale,
    contours_coloring='heatmap',
    line_smoothing=0.85,
    colorbar=dict(title="Pa" if visual_type == "שדה לחצים (Pressure)" else "m/s")
))

# 2. הוספת קווי מתווה הכנף
fig.add_trace(go.Contour(
    z=inside.astype(int),
    x=x,
    y=y,
    showscale=False,
    contours=dict(start=0.5, end=0.5, coloring='none'),
    line=dict(color='black', width=3)
))

fig.update_layout(
    title=f"CFD Simulation - AoA: {aoa}°, Speed: {velocity} m/s",
    xaxis_title="X Position (m)",
    yaxis_title="Y Position (m)",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
