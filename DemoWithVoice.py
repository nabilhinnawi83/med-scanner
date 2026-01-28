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

# --- MOBILE VOICE ENGINE ---
def text_to_speech_mobile(text):
    if text:
        clean_text = text.replace("'", "").replace("\n", " ").replace("*", "").replace('"', '')
        components_code = f"""
            <script>
            window.speechSynthesis.cancel(); 
            var msg = new SynthesisUtterance("{clean_text}");
            msg.lang = 'en-US';
            msg.rate = 1.0;
            setTimeout(function(){{
                window.speechSynthesis.speak(msg);
            }}, 100);
            </script>
        """
        st.components.v1.html(components_code, height=0)

# --- UI ---
st.title("📱 Siri Medicine Scanner")

if "siri_unlocked" not in st.session_state:
    st.session_state.siri_unlocked = False

if not st.session_state.siri_unlocked:
    st.info("Tap below to enable Siri's voice.")
    if st.button("🚀 Start Siri"):
        st.session_state.siri_unlocked = True
        text_to_speech_mobile("Siri is active.")
        st.rerun()
else:
    img_file = st.camera_input("Scan the label")

    if img_file:
        img_bytes = img_file.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        with st.spinner("🧠 Siri is analyzing..."):
            # STEP 1: Strict Quality Gateway
            gate_prompt = """
            ACT AS A STRICT MEDICINE SAFETY SCANNER. 
            Analyze the image for the medicine name.
            1. If name is cut off or hidden, set quality='INCOMPLETE'.
            2. If quality is 'INCOMPLETE', set siri_message='The name is cut off. Please show me the full label so I can be sure'.
            Return ONLY JSON: {"quality": "GOOD/INCOMPLETE", "medicine_name": "Name", "siri_message": "..."}
            """
            
            gate_payload = {
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
                res = requests.post(NIM_URL, json=gate_payload).json()
                data = json.loads(res['choices'][0]['message']['content'])
                
                # Check Quality First
                if data.get('quality') != "GOOD":
                    error_msg = data.get('siri_message', "I can't see the label clearly.")
                    st.error(error_msg)
                    text_to_speech_mobile(error_msg)
                else:
                    med_name = data.get('medicine_name')
                    st.success(f"Verified: {med_name}")

                    # STEP 2: Oracle Vector Search
                    conn = oracledb.connect(user=DB_CONFIG["user"], password=DB_CONFIG["password"], dsn=DB_CONFIG["dsn"])
                    with conn.cursor() as cursor:
                        sql = "SELECT LEAF_TEXT FROM MEDICINE_LEAFLETS ORDER BY VECTOR_DISTANCE(LEAF_VECTOR, VECTOR_EMBEDDING(MED_EMBED_MODEL USING :name AS DATA), COSINE) FETCH FIRST 1 ROWS ONLY"
                        cursor.execute(sql, name=med_name)
                        row = cursor.fetchone()
                        
                        if row:
                            leaflet_text = row[0].read() if hasattr(row[0], 'read') else row[0]
                            
                            # STEP 3: Strict 3-Sentence Output
                            summary_prompt = f"""
                            Based on this text: {leaflet_text[:2000]}
                            Provide information for {med_name} in exactly three short sentences.
                            Use this exact format:
                            Medicine is [Name]. Usage is [Short usage]. Dose is [Short dose].
                            No other text.
                            """
                            
                            sum_res = requests.post(NIM_URL, json={
                                "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1", 
                                "messages": [{"role": "user", "content": summary_prompt}]
                            }).json()
                            
                            siri_output = sum_res['choices'][0]['message']['content'].strip()
                            st.success(f"**Siri:** {siri_output}")
                            text_to_speech_mobile(siri_output)
                        else:
                            st.warning("Not found in database.")
                    conn.close()
            except Exception as e:
                st.error(f"Error: {e}")
