import streamlit as st
import base64
import requests
import json
import oracledb

# --- DATABASE CONFIG ---
NIM_URL = "http://161.33.44.233/v1/chat/completions"

DSN_STRING = """(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.eu-frankfurt-1.oraclecloud.com))(connect_data=(service_name=gf98d0d123772ee_hackathonaidb_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))"""

DB_CONFIG = {
    "user": "ADMIN",
    "password": "Team01ButNoParis!?",
    "dsn": DSN_STRING
}

st.set_page_config(page_title="Siri Med Scanner", page_icon="??")

# Custom Function for Mobile Voice (JavaScript)
def text_to_speech_mobile(text):
    if text:
        # Clean the text to prevent JS errors
        clean_text = text.replace("'", "").replace("\n", " ")
        components_code = f"""
            <script>
            var msg = new SpeechSynthesisUtterance('{clean_text}');
            window.speechSynthesis.speak(msg);
            </script>
        """
        st.components.v1.html(components_code, height=0)

st.title("?? Siri Medicine Scanner")
st.write("Scan your medicine to verify registration and dosage.")

# --- MOBILE CAMERA INPUT ---
img_file = st.camera_input("Take a picture of the label")

if img_file:
    img_bytes = img_file.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    with st.spinner("?? Siri is analyzing..."):
        # STEP 1: Quality Gate
        # Fixed Indentation here
        gate_prompt = """
        ACT AS A STRICT MEDICINE SAFETY SCANNER. 
        Analyze the image for the medicine name.
        
        CRITICAL RULES:
        1. If ANY part of the brand name is cut off, hidden, or obscured, you MUST set quality to 'INCOMPLETE'.
        2. If quality is 'INCOMPLETE' or 'BLURRY', the 'siri_message' MUST BE: "The name is cut off. Please show me the full label so I can be sure"
        3. Do not try to guess the name from fragments. 
        
        Return ONLY this JSON format:
        {
          "quality": "GOOD" or "BLURRY" or "INCOMPLETE",
          "medicine_name": "Name" or "None",
          "siri_message": "..."
        }
        """
        
        payload = {
            "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": gate_prompt}, 
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }
        
        res = requests.post(NIM_URL, json=payload).json()
        data = json.loads(res['choices'][0]['message']['content'])

        if data.get('quality') != "GOOD":
            report = data.get('siri_message', "The image is not clear. Please try again.")
            st.error(report)
            text_to_speech_mobile(report)
        else:
            med_name = data.get('medicine_name')
            st.success(f"Scanning {med_name}...")
            
            # STEP 2: Oracle Vector Search
            conn = oracledb.connect(user=DB_CONFIG["user"], password=DB_CONFIG["password"], dsn=DB_CONFIG["dsn"])
            cursor = conn.cursor()
            sql = """
                SELECT LEAF_TEXT, 
                VECTOR_DISTANCE(LEAF_VECTOR, VECTOR_EMBEDDING(MED_EMBED_MODEL USING :name AS DATA), COSINE) as dist 
                FROM MEDICINE_LEAFLETS 
                ORDER BY dist FETCH FIRST 1 ROWS ONLY
            """
            cursor.execute(sql, name=med_name)
            row = cursor.fetchone()
            conn.close()

            if row and row[1] < 0.8:
                leaflet_text = row[0].read() if hasattr(row[0], 'read') else row[0]
                
                # STEP 3: Summarize
                summary_prompt = f"ACT AS A PHARMACIST. Based on this leaflet, extract ONLY the Name, Usage, and Dose: \n{leaflet_text}"
                sum_res = requests.post(NIM_URL, json={
                    "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1", 
                    "messages": [{"role": "user", "content": summary_prompt}]
                }).json()
                
                report = sum_res['choices'][0]['message']['content']
                st.info(report)
                text_to_speech_mobile(report)
            else:
                msg = f"I see {med_name}, but it is not in our registry."
                st.warning(msg)
                text_to_speech_mobile(msg)