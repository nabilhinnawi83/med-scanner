import streamlit as st
import base64
import requests
import json
import oracledb

# --- DATABASE CONFIG ---
NIM_URL = "http://161.33.44.233/v1/chat/completions"
DSN_STRING = """(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1521)(host=adb.eu-frankfurt-1.oraclecloud.com))(connect_data=(service_name=gf98d0d123772ee_hackathonaidb_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))"""

DB_CONFIG = {
    "user": "ADMIN",
    "password": "Team01ButNoParis!?",
    "dsn": DSN_STRING
}

st.set_page_config(page_title="Siri Med Scanner", page_icon="💊", layout="centered")

# --- MOBILE VOICE ENGINE (JavaScript) ---
def text_to_speech_mobile(text):
    if text:
        # Clean text for JS safety
        clean_text = text.replace("'", "").replace("\n", " ").replace("*", "")
        components_code = f"""
            <script>
            // The 'Force Wake' Patch for Mobile
            window.speechSynthesis.cancel(); 
            var msg = new SpeechSynthesisUtterance('{clean_text}');
            msg.lang = 'en-US';
            msg.rate = 0.9;
            
            // Short delay ensures the browser processes the 'speak' command after interaction
            setTimeout(function(){{
                window.speechSynthesis.speak(msg);
            }}, 100);
            </script>
        """
        st.components.v1.html(components_code, height=0)

# --- UI HEADER ---
st.title("📱 Siri Medicine Scanner")
st.write("Scan your medicine to verify registration and dosage.")

# --- MOBILE HANDSHAKE ---
if "siri_unlocked" not in st.session_state:
    st.session_state.siri_unlocked = False

if not st.session_state.siri_unlocked:
    st.info("Welcome! Tap below to enable Siri's voice.")
    if st.button("🚀 Start Siri"):
        st.session_state.siri_unlocked = True
        text_to_speech_mobile("Siri is active and ready for your scan.")
        st.rerun()
else:
    img_file = st.camera_input("Take a clear picture of the medicine label")

    if img_file:
        img_bytes = img_file.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        with st.spinner("🧠 Siri is analyzing..."):
            gate_prompt = """
            Analyze the image for the medicine name.
            Return ONLY JSON: {"quality": "GOOD/INCOMPLETE", "medicine_name": "Name", "siri_message": "..."}
            """
            
            payload = {
                "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
                "messages": [{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": gate_prompt}, 
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }],
                "response_format": {"type": "json_object"}
            }
            
            try:
                res = requests.post(NIM_URL, json=payload).json()
                data = json.loads(res['choices'][0]['message']['content'])

                if data.get('quality') != "GOOD":
                    report = data.get('siri_message', "The label is not clear.")
                    st.error(report)
                    text_to_speech_mobile(report)
                else:
                    med_name = data.get('medicine_name')
                    st.success(f"Verified: {med_name}")
                    
                    # Oracle Connection
                    conn = oracledb.connect(user=DB_CONFIG["user"], password=DB_CONFIG["password"], dsn=DB_CONFIG["dsn"])
                    with conn.cursor() as cursor:
                        sql = "SELECT LEAF_TEXT, VECTOR_DISTANCE(LEAF_VECTOR, VECTOR_EMBEDDING(MED_EMBED_MODEL USING :name AS DATA), COSINE) as dist FROM MEDICINE_LEAFLETS ORDER BY dist FETCH FIRST 1 ROWS ONLY"
                        cursor.execute(sql, name=med_name)
                        row = cursor.fetchone()
                        
                        if row and row[1] < 0.8:
                            leaflet_text = row[0].read() if hasattr(row[0], 'read') else row[0]
                            summary_prompt = f"Extract Name, Usage, and Dose: \n{leaflet_text}"
                            sum_res = requests.post(NIM_URL, json={"model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1", "messages": [{"role": "user", "content": summary_prompt}]}).json()
                            report = sum_res['choices'][0]['message']['content']
                            st.markdown(f"### 📋 Pharmacist Report\n{report}")
                            text_to_speech_mobile(report)
                        else:
                            text_to_speech_mobile(f"Medicine {med_name} not found in registry.")
                    conn.close()

            except Exception as e:
                st.error(f"System Error: {e}")
