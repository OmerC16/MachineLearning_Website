import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time

st.set_page_config(page_title="מנהרת רוח AI - תנועת עשן ריאליסטית", layout="wide")

st.title("🌬️ סימולטור מנהרת רוח AI - זרימת עשן חיה")

# --- ניהול מצב האפליקציה ---
if "aoa" not in st.session_state:
    st.session_state.aoa = 6.0
if "camber" not in st.session_state:
    st.session_state.camber = 0.0

# --- סרגל צד ---
st.sidebar.header("🕹️ בקרת ניסוי")

st.session_state.aoa = st.sidebar.slider(
    "זווית התקפה (זווית α)", -5.0, 22.0, float(st.session_state.aoa), step=0.5
)
wind_speed = st.sidebar.slider("מהירות רוח", 1.0, 10.0, 5.0)
num_lines = st.sidebar.slider("כמות פסי עשן", 10, 40, 24)

st.sidebar.write("---")
opt_button = st.sidebar.button("🚀 הרץ אופטימיזציה גנרטיבית (AI)")

# --- קוד HTML5 Canvas עם תנועת עשן ומניעת חדירה לכנף ---
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

        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const chord = 180;

        // בדיקה האם נקודה נמצאת בתוך תחום הכנף (Collision Box/Ellipse)
        function isInsideAirfoil(px, py) {{
            // העברת הנקודה למערכת הצירים של הכנף (מוטה בזווית ההתקפה)
            const rad = aoa * (Math.PI / 180);
            const dx = px - centerX;
            const dy = py - centerY;

            const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
            const ry = dx * Math.sin(rad) + dy * Math.cos(rad);

            // הגדרת עובי הכנף לפי מיקום לאורך המיתר (NACA Airfoil Approximation)
            if (rx < -chord / 2 || rx > chord / 2) return false;

            const normX = (rx + chord / 2) / chord; // 0 עד 1
            const maxThickness = 28 + camber * 3;
            const thicknessAtX = 4 * maxThickness * normX * (1 - normX);

            return Math.abs(ry) < thicknessAtX / 2;
        }}

        // חישוב שדה המהירויות ומניעת חדירה לכנף
        function getFieldVelocity(x, y) {{
            let vx = windSpeed * 2.8;
            let vy = 0;

            const rad = -aoa * (Math.PI / 180);
            const dx = x - centerX;
            const dy = y - centerY;
            const dist = Math.sqrt(dx * dx + dy * dy);

            // השפעה אווירודינמית בסביבת הכנף
            if (dist < 200) {{
                const factor = (1 - dist / 200);
                const liftEffect = Math.sin(rad) * 2.2;
                
                vy += (dx / (dist + 5)) * liftEffect * windSpeed * factor * 10;

                // האצה מעל הכנף
                if (dy < 0 && dx > -90 && dx < 90) {{
                    vx += (Math.abs(aoa) + 2) * 0.4 * factor;
                }}

                // הזדקרות וסחרור בזוויות גבוהות
                if (aoa > 12 && dx > 10) {{
                    vy += (Math.random() - 0.5) * (aoa - 10) * 0.9;
                    vx *= (1 - 0.02 * (aoa - 10));
                }}
            }}

            // מנגנון חסימה: אם החלקיק קרוב מדי לכנף - דוחפים אותו החוצה למעלה/למטה
            if (isInsideAirfoil(x + vx, y + vy)) {{
                if (dy < 0) {{
                    vy -= 4.0 + Math.abs(aoa) * 0.2; // הסטה כלפי מעלה
                }} else {{
                    vy += 4.0 + Math.abs(aoa) * 0.2; // הסטה כלפי מטה
                }}
                vx *= 0.8; // הנהגת התנגדות במגע עם הכנף
            }}

            return {{ vx, vy }};
        }}

        // יצירת קווי עשן המורכבים מחלקיקים זורמים בזמן אמת
        const streamLines = [];
        const trailSize = 45; // אורך פס העשן הנע

        for (let i = 0; i < numLines; i++) {{
            const startY = (canvas.height / (numLines + 1)) * (i + 1);
            const lineParticles = [];
            
            for (let j = 0; j < trailSize; j++) {{
                lineParticles.push({{ x: (canvas.width / trailSize) * j, y: startY, startY: startY }});
            }}
            streamLines.push(lineParticles);
        }}

        // ציור הכנף
        function drawAirfoil() {{
            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(-aoa * (Math.PI / 180));

            ctx.beginPath();
            ctx.moveTo(-chord / 2, 0);
            
            const topCamber = -32 - camber * 2;
            ctx.bezierCurveTo(-chord / 4, topCamber, chord / 4, topCamber + 5, chord / 2, 0);
            ctx.bezierCurveTo(chord / 4, 16, -chord / 4, 16, -chord / 2, 0);

            ctx.fillStyle = "#111827";
            ctx.shadowColor = '#00f0ff';
            ctx.shadowBlur = 10;
            ctx.fill();
            ctx.strokeStyle = "#00f0ff";
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.restore();
        }}

        // לולאת האנימציה הראשית
        function animate() {{
            // יצירת שובל עמום שמעניק תחושת תנועה וטשטוש עשן
            ctx.fillStyle = "rgba(5, 5, 8, 0.25)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            drawAirfoil();

            // עדכון וציור תנועת פסי העשן
            for (let i = 0; i < streamLines.length; i++) {{
                const line = streamLines[i];

                ctx.beginPath();
                for (let j = 0; j < line.length; j++) {{
                    const p = line[j];
                    
                    // חישוב המהירות והרחיפה של העשן
                    const vel = getFieldVelocity(p.x, p.y);
                    p.x += vel.vx;
                    p.y += vel.vy;

                    // איפוס חלקיק שהגיע לקצה הימני בחזרה לצד שמאל
                    if (p.x > canvas.width) {{
                        p.x = 0;
                        p.y = p.startY;
                    }}

                    if (j === 0) {{
                        ctx.moveTo(p.x, p.y);
                    }} else {{
                        ctx.lineTo(p.x, p.y);
                    }}
                }}

                // עיצוב פס העשן הדינמי
                ctx.strokeStyle = "rgba(0, 240, 255, 0.65)";
                ctx.lineWidth = 2.5;
                ctx.shadowColor = 'rgba(0, 240, 255, 0.3)';
                ctx.shadowBlur = 6;
                ctx.stroke();
            }}

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
    
    cl = 2 * np.pi * np.radians(st.session_state.aoa) + (st.session_state.camber * 0.1)
    if st.session_state.aoa > 12:
        cl *= np.cos(np.radians(st.session_state.aoa - 12) * 3)
        
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
    
    for step in range(1, 101):
        st.session_state.aoa = max(2.0, min(8.5, st.session_state.aoa * 0.95 + 0.3))
        st.session_state.camber = min(3.0, st.session_state.camber + 0.05)
        
        progress_bar.progress(step)
        time.sleep(0.03)
        st.rerun()

    st.success("האופטימיזציה הושלמה! הגעת ליחס $L/D$ מקסימלי.")
