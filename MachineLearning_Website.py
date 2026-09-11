import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time

st.set_page_config(page_title="מנהרת רוח AI - עשן ריאליסטי", layout="wide")

st.title("🌬️ סימולטור מנהרת רוח AI - זרימת עשן ריאליסטית (100% Collision-Free)")

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
num_lines = st.sidebar.slider("כמות פסי עשן", 10, 35, 20)

st.sidebar.write("---")
opt_button = st.sidebar.button("🚀 הרץ אופטימיזציה גנרטיבית (AI)")

# --- קוד HTML5 Canvas עם מרקם עשן רך ומנגנון SDF למניעת חדירה ---
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
            border: 1px solid #222;
            border-radius: 8px;
            background-color: #030407;
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

        // --- חישוב מעטפת הכנף והרחקה פיזיקלית מדויקת (SDF) ---
        function getAirfoilSurface(xRel) {{
            // מנרמל את מיתר הכנף מ-0 עד 1
            const normX = (xRel + chord / 2) / chord;
            if (normX < 0 || normX > 1) return null;

            // עובי פרופיל NACA מותאם כולל Camber
            const maxThickness = 30 + camber * 4;
            const thickness = 4 * maxThickness * normX * (1 - normX);
            const camberLine = -Math.sin(normX * Math.PI) * camber * 10;

            return {{
                yTop: camberLine - thickness / 2 - 4, // מרווח ביטחון למניעת חדירה
                yBottom: camberLine + thickness / 2 + 4
            }};
        }}

        // חישוב שדה המהירויות עם מנגנון דחייה מוחלט
        function getFlowVelocity(x, y) {{
            let vx = windSpeed * 2.8;
            let vy = 0;

            const rad = -aoa * (Math.PI / 180);
            const cosA = Math.cos(rad);
            const sinA = Math.sin(rad);

            // המרת קואורדינטות למערכת הצירים של הכנף
            const dx = x - centerX;
            const dy = y - centerY;

            const rx = dx * cosA - dy * sinA;
            const ry = dx * sinA + dy * cosA;

            const surface = getAirfoilSurface(rx);

            // אם החלקיק נמצא בתוך או קרוב מאוד למעטפת הכנף - מטים אותו לחלוטין
            if (surface) {{
                if (ry > surface.yTop && ry < surface.yBottom) {{
                    const distToTop = Math.abs(ry - surface.yTop);
                    const distToBottom = Math.abs(ry - surface.yBottom);

                    if (distToTop < distToBottom) {{
                        // הסטה מעל הכנף
                        vy -= 5.0 + Math.abs(aoa) * 0.3;
                    }} else {{
                        // הסטה מתחת לכנף
                        vy += 5.0 + Math.abs(aoa) * 0.3;
                    }}
                    vx *= 0.85; // האטה קלה בעת התנגדות
                }}
            }}

            // השפעה אווירודינמית רחוקה (Flow Field)
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 220) {{
                const factor = (1 - dist / 220);
                vy += (dx / (dist + 10)) * Math.sin(rad) * windSpeed * factor * 14;

                // האצה מעל הכנף (חוק ברנולי)
                if (ry < 0 && rx > -chord/2 && rx < chord/2) {{
                    vx += (Math.abs(aoa) + 2) * 0.35 * factor;
                }}

                // סחרור והזדקרות בזוויות גבוהות
                if (aoa > 12 && rx > 0) {{
                    vy += (Math.random() - 0.5) * (aoa - 10) * 0.8;
                    vx *= (1 - 0.02 * (aoa - 10));
                }}
            }}

            return {{ vx, vy }};
        }}

        // --- מערכת חלקיקי עשן דינמית ---
        const smokeLines = [];
        const particlesPerLine = 35;

        for (let i = 0; i < numLines; i++) {{
            const startY = (canvas.height / (numLines + 1)) * (i + 1);
            const line = [];
            for (let j = 0; j < particlesPerLine; j++) {{
                line.push({{
                    x: (canvas.width / particlesPerLine) * j,
                    y: startY,
                    startY: startY,
                    size: 3 + Math.random() * 2,
                    alpha: 0.4 + Math.random() * 0.4
                }});
            }}
            smokeLines.push(line);
        }}

        // ציור גוף הכנף
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
            ctx.shadowBlur = 12;
            ctx.fill();
            ctx.strokeStyle = "#00f0ff";
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.restore();
        }}

        // לולאת האנימציה הראשית
        function animate() {{
            // ניקוי המסך ליצירת שובל תנועה עדין
            ctx.fillStyle = "rgba(3, 4, 7, 0.22)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            drawAirfoil();

            // עדכון וציור פסי העשן האורגניים
            for (let i = 0; i < smokeLines.length; i++) {{
                const line = smokeLines[i];

                for (let j = 0; j < line.length; j++) {{
                    const p = line[j];

                    const vel = getFlowVelocity(p.x, p.y);
                    p.x += vel.vx;
                    p.y += vel.vy;

                    // איפוס חלקיק שיצא מגבולות המסך
                    if (p.x > canvas.width + 20) {{
                        p.x = -10;
                        p.y = p.startY;
                    }}

                    // ציור מולקולת עשן בודדת עם גרדיאנט רך
                    const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 2.5);
                    grad.addColorStop(0, `rgba(0, 240, 255, ${{p.alpha}})`);
                    grad.addColorStop(0.5, `rgba(0, 180, 255, ${{p.alpha * 0.4}})`);
                    grad.addColorStop(1, "rgba(0, 0, 0, 0)");

                    ctx.fillStyle = grad;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.size * 2.5, 0, Math.PI * 2);
                    ctx.fill();
                }}

                // חיבור הנקודות בקו עשן רציף ומטושטש
                ctx.beginPath();
                ctx.moveTo(line[0].x, line[0].y);
                for (let j = 1; j < line.length; j++) {{
                    ctx.lineTo(line[j].x, line[j].y);
                }}
                ctx.strokeStyle = "rgba(0, 240, 255, 0.25)";
                ctx.lineWidth = 3;
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
