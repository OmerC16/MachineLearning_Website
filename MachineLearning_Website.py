import streamlit as st
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="מנהרת רוח - זרימת עשן חלקת", layout="wide")

st.title("🌬️ סימולטור מנהרת רוח בזמן אמת - קווי עשן (60 FPS)")

# --- סרגל צד לפרמטרים ---
st.sidebar.header("פרמטרי ניסוי")
aoa = st.sidebar.slider("זווית התקפה (זווית α)", -5, 22, 8)
wind_speed = st.sidebar.slider("מהירות רוח", 1.0, 10.0, 4.0)
num_lines = st.sidebar.slider("כמות פסי עשן", 10, 50, 25)
line_thickness = st.sidebar.slider("עובי פסי העשן", 1, 5, 2)

# --- קוד HTML5 + JavaScript לאנימציה חלקה בדפדפן ---
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
    <canvas id="windTunnel" width="900" height="450"></canvas>

    <script>
        const canvas = document.getElementById('windTunnel');
        const ctx = canvas.getContext('2d');

        // פרמטרים מ-Python
        const aoa = {aoa};
        const windSpeed = {wind_speed};
        const numLines = {num_lines};
        const lineWidth = {line_thickness};

        // הגדרת פסי העשן (Streamlines)
        const particles = [];
        const trailLength = 25; // אורך פס העשן (מספר מקטעים)

        // יצירת נקודות התחלה לפסי העשן
        for (let i = 0; i < numLines; i++) {{
            const startY = (canvas.height / (numLines + 1)) * (i + 1);
            const history = [];
            for (let j = 0; j < trailLength; j++) {{
                history.push({{ x: -j * 10, y: startY }});
            }}
            particles.push({{
                x: Math.random() * canvas.width,
                y: startY,
                startY: startY,
                history: history
            }});
        }}

        // פונקציה לחישוב שדה המהירויות סביב הכנף (שילוב חישוב פיזיקלי/AI)
        function getVelocity(px, py) {{
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            // המרה לקואורדינטות יחסיות לכנף
            const dx = px - centerX;
            const dy = py - centerY;
            
            let vx = windSpeed * 2;
            let vy = 0;

            // מרחק ממרכז הכנף
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < 180) {{
                const rad = -aoa * (Math.PI / 180);
                
                // הסטת הזרימה למעלה/למטה לפי זווית ההתקפה
                const liftFactor = Math.sin(rad) * 1.5;
                vy += (dx / dist) * liftFactor * windSpeed;

                // האצה מעל הכנף (חוק ברנולי)
                if (dy < 0 && dx > -80 && dx < 80) {{
                    vx += (Math.abs(aoa) + 2) * 0.4;
                    vy -= Math.abs(aoa) * 0.15;
                }}

                // אפקט הזדקרות וסחרור (Stall) בזוויות גבוהות
                if (aoa > 12 && dx > 0 && dy < 20) {{
                    vy += (Math.random() - 0.5) * (aoa - 10) * 0.8;
                    vx *= 0.6; // ירידה במהירות באזור הסחרור
                }}
            }}

            return {{ vx, vy }};
        }}

        // ציור הכנף בתוך ה-Canvas
        function drawAirfoil() {{
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(-aoa * (Math.PI / 180));

            ctx.beginPath();
            // פרופיל כנף NACA 0012 מוגדל
            const chord = 160;
            ctx.moveTo(-chord / 2, 0);
            
            ctx.bezierCurveTo(
                -chord / 4, -30, 
                 chord / 4, -25, 
                 chord / 2, 0
            );
            ctx.bezierCurveTo(
                 chord / 4, 15, 
                -chord / 4, 15, 
                -chord / 2, 0
            );

            ctx.fillStyle = "#e0e0e0";
            ctx.shadowColor = 'cyan';
            ctx.shadowBlur = 10;
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.restore();
        }}

        // לולאת האנימציה הראשית (60 FPS)
        function animate() {{
            // ניקוי המסך עם אפקט שובל קל
            ctx.fillStyle = "rgba(5, 5, 8, 0.3)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            drawAirfoil();

            // עדכון וציור פסי העשן
            for (let i = 0; i < particles.length; i++) {{
                const p = particles[i];

                // חישוב המהירות בנקודה הנוכחית
                const vel = getVelocity(p.x, p.y);
                p.x += vel.vx;
                p.y += vel.vy;

                // עדכון היסטוריית הנקודות ליצירת פס רציף
                p.history.push({{ x: p.x, y: p.y }});
                if (p.history.length > trailLength) {{
                    p.history.shift();
                }}

                // אם הפס יצא מגבולות המסך - מחזירים אותו להתחלה
                if (p.x > canvas.width + 50) {{
                    p.x = -50;
                    p.y = p.startY;
                    p.history = [];
                    for (let j = 0; j < trailLength; j++) {{
                        p.history.push({{ x: -50, y: p.startY }});
                    }}
                }}

                // ציור פס העשן כקו מעוגל ורציף
                ctx.beginPath();
                ctx.lineWidth = lineWidth;
                
                for (let j = 0; j < p.history.length - 1; j++) {{
                    const pt1 = p.history[j];
                    const pt2 = p.history[j + 1];
                    
                    // שקיפות מדורגת (Fade Out) בזנב הפס
                    const alpha = (j / p.history.length);
                    ctx.strokeStyle = `rgba(0, 240, 255, ${{alpha}})`;
                    
                    ctx.beginPath();
                    ctx.moveTo(pt1.x, pt1.y);
                    ctx.lineTo(pt2.x, pt2.y);
                    ctx.stroke();
                }}
            }}

            requestAnimationFrame(animate);
        }}

        animate();
    </script>
</body>
</html>
"""

# הצגת רכיב האנימציה ב-Streamlit
components.html(canvas_code, height=480)
