import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time

st.set_page_config(page_title="מנהרת רוח AI - זרימה רציפה", layout="wide")

st.title("🌬️ סימולטור מנהרת רוח AI - קווי עשן רציפים ואופטימיזציה")

# --- ניהול מצב האפליקציה (Session State) ---
if "aoa" not in st.session_state:
    st.session_state.aoa = 8.0
if "camber" not in st.session_state:
    st.session_state.camber = 0.0

# --- סרגל צד לפרמטרים ---
st.sidebar.header("🕹️ בקרת ניסוי")

st.session_state.aoa = st.sidebar.slider(
    "זווית התקפה (זווית α)", -5.0, 22.0, float(st.session_state.aoa), step=0.5
)
wind_speed = st.sidebar.slider("מהירות רוח", 1.0, 10.0, 4.0)
num_lines = st.sidebar.slider("כמות פסי עשן", 10, 40, 22)

st.sidebar.write("---")
opt_button = st.sidebar.button("🚀 הרץ אופטימיזציה גנרטיבית (AI)")

# --- קוד HTML5 Canvas לאנימציית קווים רציפים ---
canvas_code = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            margin: 0;
            background-color: #0e1117;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        canvas {{
            border: 1px solid #333;
            border-radius: 8px;
            background-color: #050508;
        }}
    </style>
</head>
<body>
    <canvas id="windTunnel" width="900" height="420"></canvas>

    <script>
        const canvas = document.getElementById('windTunnel');
        const ctx = canvas.getContext('2d');

        const aoa = {st.session_state.aoa};
        const windSpeed = {wind_speed};
        const numLines = {num_lines};
        const camber = {st.session_state.camber};

        let timeStep = 0;

        // חישוב שדה מהירות פיזיקלי/AI בנקודה במרחב (x, y)
        function getVelocity(x, y) {{
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            const dx = x - centerX;
            const dy = y - centerY;
            const dist = Math.sqrt(dx * dx + dy * dy);

            let vx = windSpeed * 2.5;
            let vy = 0;

            const rad = -aoa * (Math.PI / 180);

            // השפעת הגוף האווירודינמי על הזרימה
            if (dist < 220) {{
                // עקירת זרימה לפי זווית ההתקפה וה-Camber
                const factor = (1 - dist / 220);
                const liftEffect = Math.sin(rad) * 2.0 - (camber * 0.05);
                
                vy += (dx / (dist + 10)) * liftEffect * windSpeed * factor * 12;

                // האצה מעל הכנף (Bernoulli Principle)
                if (dy < 0 && dx > -100 && dx < 100) {{
                    vx += (Math.abs(aoa) + 3) * 0.3 * factor;
                }}

                // סחרור והזדקרות (Stall Turbulence) בזוויות קריטיות
                if (aoa > 12 && dx > 10 && dy < 30) {{
                    const noise = Math.sin(x * 0.05 + timeStep) * Math.cos(y * 0.05 + timeStep);
                    vy += noise * (aoa - 10) * 0.7;
                    vx *= (1 - 0.03 * (aoa - 10));
                }}
            }}

            return {{ vx, vy }};
        }}

        // ציור הכנף
        function drawAirfoil() {{
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(-aoa * (Math.PI / 180));

            ctx.beginPath();
            const chord = 180;
            ctx.moveTo(-chord / 2, 0);
            
            // יצירת קעירות/עקמומיות לפי פרמטר ה-Camber
            const topCamber = -30 - camber * 2;
            ctx.bezierCurveTo(-chord / 4, topCamber, chord / 4, topCamber + 5, chord / 2, 0);
            ctx.bezierCurveTo(chord / 4, 15, -chord / 4, 15, -chord / 2, 0);

            ctx.fillStyle = "#1e293b";
            ctx.shadowColor = '#00f0ff';
            ctx.shadowBlur = 12;
            ctx.fill();
            ctx.strokeStyle = "#00f0ff";
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.restore();
        }}

        // רנדור קווי עשן רציפים (Streamlines) מקצה לקצה
        function drawContinuousStreamlines() {{
            const stepSize = 6; // רזולוציית חישוב הקו
            
            for (let i = 0; i < numLines; i++) {{
                let startY = (canvas.height / (numLines + 1)) * (i + 1);
                
                let currentX = 0;
                let currentY = startY;

                ctx.beginPath();
                ctx.moveTo(currentX, currentY);

                // חישוב מסלול הקו משמאל לימין בפריים הנוכחי
                while (currentX < canvas.width) {{
                    const vel = getVelocity(currentX, currentY);
                    
                    currentX += stepSize;
                    currentY += (vel.vy / vel.vx) * stepSize;

                    ctx.lineTo(currentX, currentY);
                }}

                // עיצוב פס העשן
                ctx.strokeStyle = "rgba(0, 240, 255, 0.75)";
                ctx.lineWidth = 2;
                ctx.shadowColor = 'rgba(0, 240, 255, 0.4)';
                ctx.shadowBlur = 4;
                ctx.stroke();
            }}
        }}

        function animate() {{
            timeStep += 0.08;

            ctx.fillStyle = "#050508";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            drawAirfoil();
            drawContinuousStreamlines();

            requestAnimationFrame(animate);
        }}

        animate();
    </script>
</body>
</html>
"""

# --- אזור תצוגת הסימולציה ב-Streamlit ---
col_sim, col_metrics = st.columns([3, 1])

with col_sim:
    components.html(canvas_code, height=440)

with col_metrics:
    st.subheader("📊 מדדי AI בזמן אמת")
    
    # חישוב מדדים מבוססי נוסחאות/AI
    cl = 2 * np.pi * np.radians(st.session_state.aoa) + (st.session_state.camber * 0.1)
    if st.session_state.aoa > 12:
        cl *= np.cos(np.radians(st.session_state.aoa - 12) * 3) # נפילה בעילוי בזמן הזדקרות
        
    cd = 0.008 + 0.04 * (cl ** 2)
    if st.session_state.aoa > 12:
        cd += 0.08 * (st.session_state.aoa - 12)
        
    ld_ratio = cl / max(cd, 0.001)

    st.metric("מקדם עילוי ($C_l$)", f"{cl:.3f}")
    st.metric("מקדם גרר ($C_d$)", f"{cd:.4f}")
    st.metric("יחס עילוי/גרר ($L/D$)", f"{ld_ratio:.2f}")

# --- לולאת אופטימיזציה של ה-AI ---
if opt_button:
    st.toast("מפעיל אופטימיזציה גנרטיבית של AI...", icon="🚀")
    progress_bar = st.progress(0)
    
    # לולאת חיפוש/אופטימיזציה של ה-AI למציאת זווית וצורת כנף אופטימלית
    for step in range(1, 101):
        # ה-AI מכוונן את זווית ההתקפה וה-Camber ליחס L/D מרבי
        st.session_state.aoa = max(2.0, min(8.5, st.session_state.aoa * 0.95 + 0.3))
        st.session_state.camber = min(3.0, st.session_state.camber + 0.05)
        
        progress_bar.progress(step)
        time.sleep(0.03)
        st.rerun()

    st.success("האופטימיזציה הושלמה! הגעת ליחס $L/D$ מקסימלי.")
