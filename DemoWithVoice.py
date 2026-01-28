import streamlit as st
import base64
import requests
import json
import oracledb

# --- DATABASE CONFIG ---
NIM_URL = "http://161.33.44.233/v1/chat/completions"

# Port 1521 is required for TLS (Walletless) when mTLS is "Not Required"
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
            // Force reset of audio engine for mobile browsers
            window.speechSynthesis.cancel(); 
            var msg = new SpeechSynthesisUtterance('{clean_text}');
            msg.lang = 'en-US';
            msg.rate = 0.9;
            
            // Short delay helps mobile browsers process the speech request
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
# Mobile browsers block audio until the user interacts with the page
if "siri_unlocked" not in st.session_state:
    st.session_state.siri_unlocked = False

if not st.session_state.siri_unlocked:
    st.info("Welcome! Tap the button below to enable Siri's voice.")
    if st.button("🚀 Start Siri"):
        st.session_state.siri_unlocked = True
        text_to_speech_mobile("Siri is active. Ready to scan your medicine.")
        st.rerun()
else:
    # --- CAMERA INPUT ---
    img_file = st.camera_input("Take a clear picture of the medicine label")

    if img_file:
        img_bytes = img_file.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        with st.spinner("🧠 Siri is analyzing..."):
            # STEP 1: Vision Gate (Llama-3.1-Nemotron-Nano)
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
                    report = data.get('siri_message', "The image is not clear enough.")
                    st.error(report)
                    text_to_speech_mobile(report)
                else:
                    med_name = data.get('medicine_name')
                    st.success(f"Verified: {med_name}")
                    
                    # STEP 2: Oracle Vector Search (RAG)
                    conn = oracledb.connect(user=DB_CONFIG["user"], password=DB_CONFIG["password"], dsn=DB_CONFIG["dsn"])
                    with conn.cursor() as cursor:
                        st.info(f"🔎 Checking registry for {med_name}...")
                        
                        sql = """
                            SELECT LEAF_TEXT, 
                            VECTOR_DISTANCE(LEAF_VECTOR, VECTOR_EMBEDDING(MED_EMBED_MODEL USING :name AS DATA), COSINE) as dist 
                            FROM MEDICINE_LEAFLETS 
                            ORDER BY dist FETCH FIRST 1 ROWS ONLY
                        """
                        cursor.execute(sql, name=med_name)
                        row = cursor.fetchone()
                        
                        if row and row[1] < 0.8:
                            leaflet_text = row[0].read() if hasattr(row[0], 'read') else row[0]
                            
                            # STEP 3: Verification & Summary (Pharmacist Prompt)
                            summary_prompt = f"""
                            SYSTEM: YOU ARE A LICENSED PHARMACIST.
                            USER SCANNED: {med_name}
                            DATABASE LEAFLET: {leaflet_text[:2000]} 

                            TASK:
                            1. Does this database leaflet actually belong to {med_name}? 
                            2. If YES, provide: Name, Usage, and Dose.
                            3. If NO, say: 'I found a record, but it does not match this medicine exactly. Please consult a human pharmacist.'
                            """
                            
                            sum_res = requests.post(NIM_URL, json={
                                "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1", 
                                "messages": [{"role": "user", "content": summary_prompt}]
                            }).json()
                            
                            report = sum_res['choices'][0]['message']['content']
                            st.markdown(f"### 📋 Pharmacist Report\n{report}")
                            text_to_speech_mobile(report)
                        else:
                            msg = f"I recognize {med_name}, but it is not found in our database."
                            st.warning(msg)
                            text_to_speech_mobile(msg)
                    
                    conn.close()

            except Exception as e:
                st.error(f"System Error: {e}")
