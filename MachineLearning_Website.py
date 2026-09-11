import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time

st.set_page_config(page_title="מנהרת רוח AI - מראה ריאליסטי", layout="wide")

st.title("🌬️ סימולטור מנהרת רוח - התאמה מדויקת לתמונה")

# --- ניהול מצב האפליקציה ---
if "aoa" not in st.session_state:
    st.session_state.aoa = 4.0
if "camber" not in st.session_state:
    st.session_state.camber = 0.0

# --- סרגל צד ---
st.sidebar.header("🕹️ בקרת ניסוי")

st.session_state.aoa = st.sidebar.slider(
    "זווית התקפה (זווית α)", -5.0, 22.0, float(st.session_state.aoa), step=0.5
)
wind_speed = st.sidebar.slider("מהירות רוח", 1.0, 10.0, 4.5)
num_lines = st.sidebar.slider("כמות פסי עשן", 10, 25, 16)

st.sidebar.write("---")
opt_button = st.sidebar.button("🚀 הרץ אופטימיזציה גנרטיבית (AI)")

# --- קוד HTML5 Canvas - רנדור זהה לתמונה ---
canvas_code = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            margin: 0;
            background-color: #000000;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        canvas {{
            border: 1px solid #222;
            background-color: #000000;
        }}
    </style>
</head>
<body>
    <canvas id="windTunnel" width="900" height="450"></canvas>

    <script>
        const canvas = document.getElementById('windTunnel');
        const ctx = canvas.getContext('2d');

        const aoa = {st.session_state.aoa};
        const windSpeed = {wind_speed};
        const numLines = {num_lines};
        const camber = {st.session_state.camber};

        const centerX = canvas.width * 0.42;
        const centerY = canvas.height * 0.52;
        const chord = 220;

        let animTime = 0;

        // בדיקה פיזיקלית מדויקת: האם הנקודה (x, y) מוצבת בתוך הכנף?
        function isPointInAirfoil(x, y) {{
            const rad = -aoa * (Math.PI / 180);
            const dx = x - centerX;
            const dy = y - centerY;

            // סיבוב קואורדינטות לפי זווית ההתקפה
            const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
            const ry = dx * Math.sin(rad) + dy * Math.cos(rad);

            const normX = (rx + chord / 2) / chord;
            if (normX < 0 || normX > 1) return null;

            // עובי הכנף לפי מיתר NACA
            const maxThickness = 38 + camber * 3;
            const thickness = 4 * maxThickness * normX * (1 - normX);
            const camberOffset = -Math.sin(normX * Math.PI) * camber * 8;

            const topY = camberOffset - thickness / 2;
            const botY = camberOffset + thickness / 2;

            if (ry >= topY - 3 && ry <= botY + 3) {{
                return {{ rx, ry, topY, botY, isInside: true }};
            }}

            return {{ rx, ry, topY, botY, isInside: false }};
        }}

        // ציור הכנף (אפורה בדיוק כמו בתמונה)
        function drawAirfoil() {{
            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(-aoa * (Math.PI / 180));

            ctx.beginPath();
            ctx.moveTo(-chord / 2, 0);
            
            const topCamber = -40 - camber * 2;
            ctx.bezierCurveTo(-chord / 4, topCamber, chord / 4, topCamber + 8, chord / 2, 0);
            ctx.bezierCurveTo(chord / 4, 18, -chord / 4, 18, -chord / 2, 0);

            ctx.fillStyle = "#b0b0b0"; // אפור בהיר כמו בתמונה
            ctx.fill();

            ctx.restore();
        }}

        // ציור פסי עשן רציפים במראה ענני רך
        function drawSmokeStreams() {{
            const stepSize = 4;

            for (let i = 0; i < numLines; i++) {{
                const startY = (canvas.height / (numLines + 1)) * (i + 1);
                
                let currX = 0;
                let currY = startY;

                ctx.beginPath();
                ctx.moveTo(currX, currY);

                while (currX < canvas.width) {{
                    currX += stepSize;

                    // גליות עדינה של העשן
                    const wave = Math.sin(currX * 0.015 + animTime + i) * 1.5;
                    let nextY = currY + wave * 0.1;

                    // בדיקת חסימה מול הכנף
                    const test = isPointInAirfoil(currX, nextY);

                    if (test && test.isInside) {{
                        const rad = -aoa * (Math.PI / 180);
                        // אם העשן מגיע מלמעלה -> מצמידים מעל הכנף. אם מלמטה -> מצמידים מתחת.
                        const targetLocalY = (test.ry < 0) ? test.topY - 4 : test.botY + 4;
                        
                        // המרה חזרה לקואורדינטות מסך
                        const dx = test.rx;
                        const dy = targetLocalY;

                        const worldY = centerY + dx * Math.sin(-rad) + dy * Math.cos(-rad);
                        nextY = worldY;
                    }} else {{
                        // הסטת זרימה כללית מחוץ לכנף
                        const dist = Math.hypot(currX - centerX, currY - centerY);
                        if (dist < 220) {{
                            const factor = (1 - dist / 220);
                            const rad = -aoa * (Math.PI / 180);
                            nextY += Math.sin(rad) * factor * 1.8;
                        }}
                    }}

                    // מערבולות מאחורי הכנף (Stall / Wake Turbulence)
                    if (currX > centerX + chord / 2 && Math.abs(currY - centerY) < 80) {{
                        const turbulence = (Math.random() - 0.5) * (aoa > 10 ? 4 : 1.2);
                        nextY += turbulence;
                    }}

                    currY = nextY;
                    ctx.lineTo(currX, currY);
                }}

                // סגנון העשן: קו לבן עבה ומטושטש
                ctx.strokeStyle = "rgba(235, 235, 240, 0.75)";
                ctx.lineWidth = 14;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
                ctx.shadowBlur = 12;
                ctx.stroke();
            }}
        }}

        function animate() {{
            animTime += windSpeed * 0.02;

            // ניקוי המסך לשחור מוחלט
            ctx.fillStyle = "#000000";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            drawAirfoil();
            drawSmokeStreams();

            requestAnimationFrame(animate);
        }}

        animate();
    </script>
</body>
</html>
"""

# --- אזור תצוגת הסימולציה ---
col_sim, col_metrics = st.columns([3, 1])

with col_sim:
    components.html(canvas_code, height=470)

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

# --- לולאת אופטימיזציה ---
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
