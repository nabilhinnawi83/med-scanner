import oracledb
import PyPDF2
import sys
import os

# --- CONFIGURATION ---
DB_CONFIG = {
    "user": "ADMIN",
    "password": "Team01ButNoParis!?",
    "dsn": "(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.eu-frankfurt-1.oraclecloud.com))(connect_data=(service_name=gf98d0d123772ee_hackathonaidb_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))" # Found in your OCI console or tnsnames.ora
}

def load_pdf_to_oracle(pdf_path, medicine_name, source_id):
    try:
        # 1. Connect to your Autonomous Database
        conn = oracledb.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        if not os.path.exists(pdf_path):
            print(f"? Error: File {pdf_path} not found.")
            return

        file_name = os.path.basename(pdf_path)

        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            print(f"?? Processing {medicine_name} ({len(reader.pages)} pages)...")
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    continue 
                
                # 2. MATCHED TO YOUR COLUMNS: SOURCE_ID, FILE_NAME, LEAF_TEXT, LEAF_VECTOR
                sql = """
                    INSERT INTO medicine_leaflets (SOURCE_ID, FILE_NAME, LEAF_TEXT, LEAF_VECTOR)
                    VALUES (:s_id, :f_name, :txt, VECTOR_EMBEDDING(MED_EMBED_MODEL USING :txt AS DATA))
                """
                cursor.execute(sql, s_id=source_id, f_name=file_name, txt=text)
        
        conn.commit()
        print(f"? Success! {medicine_name} (ID: {source_id}) is now vectorized.")

    except Exception as e:
        print(f"? Database Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python load_medicine.py <path_to_pdf> <medicine_name> <source_id>")
    else:
        # Example: python load_medicine.py "aspirin.pdf" "Aspirin" 101
        load_pdf_to_oracle(sys.argv[1], sys.argv[2], sys.argv[3])