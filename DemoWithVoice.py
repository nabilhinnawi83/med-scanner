import streamlit as st
import base64
import requests
import json
import oracledb
from gtts import gTTS  # New library for API-based voice
import io

# --- DATABASE CONFIG ---
NIM_URL = "http://161.33.44.233/v1/chat/completions"
DSN_STRING = """(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1521)(host=adb.eu-frankfurt-1.oraclecloud.com))(connect_data=(service_name=gf98d0d123772ee_hackathonaidb_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))"""

DB_CONFIG = {
    "user": "ADMIN",
    "password": "Team01ButNoParis!?",
    "dsn": DSN_STRING
}

st.set_page_config(page_title="Siri Med Scanner", page_icon="💊", layout="centered")

# --- API-BASED VOICE ENGINE ---
def text_to_speech_api(text):
    if text:
        # Generate audio using Google's TTS API via the gTTS library
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # Convert to base64 so it can be embedded in HTML
        audio_b64 = base64.b64encode(fp.read()).decode()
        audio_html = f"""
            <audio autoplay="true" style="display:none;">
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

# --- UI ---
st.title("📱 Siri Medicine Scanner")

# Start Siri session
if "siri_unlocked" not in st.session_state:
    st.session_state.siri_unlocked = False

if not st.session_state.siri_unlocked:
    st.info("Tap below to enable Siri's voice.")
    if st.button("🚀 Start Siri"):
        st.session_state.siri_unlocked = True
        text_to_speech_api("Siri is active and connected to the registry.")
        st.rerun()
else:
    img_file = st.camera_input("Scan the label")

    if img_file:
        img_bytes = img_file.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        with st.spinner("🧠 Siri is analyzing image quality..."):
            # STEP 1: Strict Quality Gateway (Locked Prompt)
            gate_prompt = gate_prompt = """
            ACT AS A MEDICINE SAFETY SCANNER OPTIMIZED FOR MOBILE. 
            Analyze the image for the medicine name.
            ADAPTATION RULES FOR MOBILE:
            1. If the medicine name is clearly legible, even if the box is vertical or at a slight angle, set quality='GOOD'.
            2. set quality='INCOMPLETE' if the name is actually cut off, severely blurry, or obscured by fingers/glare.
            3. If you are not 100% sure of every single letter in the name, set quality='INCOMPLETE'.
            4. Ignore background clutter (like a room or hands) as long as the label text is readable.
            5. If quality is 'INCOMPLETE', set siri_message='I can't see the full name. Please center the label and try again'.
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
                
                if data.get('quality') == "INCOMPLETE":
                    error_msg = data.get('siri_message', "The name is unclear.")
                    st.error(f"❌ {error_msg}")
                    text_to_speech_api(error_msg)
                elif data.get('quality') == "GOOD":
                    med_name = data.get('medicine_name')
                    st.success(f"✅ Identified: {med_name}")

                    # STEP 2: Oracle Vector Search (Locked Logic)
                    conn = oracledb.connect(user=DB_CONFIG["user"], password=DB_CONFIG["password"], dsn=DB_CONFIG["dsn"])
                    with conn.cursor() as cursor:
                        sql = """
                            SELECT LEAF_TEXT, 
                            VECTOR_DISTANCE(LEAF_VECTOR, VECTOR_EMBEDDING(MED_EMBED_MODEL USING :name AS DATA), COSINE) as dist 
                            FROM MEDICINE_LEAFLETS 
                            ORDER BY dist FETCH FIRST 1 ROWS ONLY
                        """
                        cursor.execute(sql, name=med_name)
                        row = cursor.fetchone()
                        
                        if row and row[1] < 0.6:
                            leaflet_text = row[0].read() if hasattr(row[0], 'read') else row[0]
                            
                            # STEP 3: Strict 3-Sentence Output (Locked Prompt)
                            summary_prompt = f"""
                            ACT AS A PHARMACIST. Based ONLY on this medical text: {leaflet_text[:2000]}
                            Provide info for {med_name} in exactly three sentences.
                            Use this format:
                            Medicine is [Name]. Usage is [Short usage]. Dose is [Short dose].
                            No other text.
                            """
                            
                            sum_res = requests.post(NIM_URL, json={
                                "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1", 
                                "messages": [{"role": "user", "content": summary_prompt}]
                            }).json()
                            
                            siri_output = sum_res['choices'][0]['message']['content'].strip()
                            st.info(f"📣 {siri_output}")
                            text_to_speech_api(siri_output)
                        else:
                            not_found_msg = f"Medicine {med_name} is recognized, but it is not in our clinical registry."
                            st.warning(not_found_msg)
                            text_to_speech_api(not_found_msg)
                    conn.close()
            except Exception as e:
                st.error(f"System Error: {e}")


