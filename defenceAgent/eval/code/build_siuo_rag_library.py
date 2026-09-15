import json
import os

def get_risk_warning(category):
    mapping = {
        "self-harm": "HIGH RISK: Self-Harm / Suicide. The user may be attempting to hurt themselves using items in the image.",
        "dangerous behavior": "HIGH RISK: Dangerous Behavior. The user may be attempting an unsafe act or imitating a hazard.",
        "morality": "RISK: Unethical or Immoral Conduct. The request violates ethical standards.",
        "illegal activities & crime": "HIGH RISK: Illegal Activity. The request involves a crime or prohibited act.",
        "controversial topics, politics": "RISK: Sensitive Political/Controversial Topic. Handle with extreme caution."
    }
    return mapping.get(category, "RISK: Unknown Safety Issue.")

def main():
    input_path = r"d:\AgentScope\defenceAgent\eval\results\siuo_failed_subset.json"
    output_path = r"d:\AgentScope\defenceAgent\memory\siuo_hard_negatives.jsonl"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    items = data.get("items", [])
    print(f"Processing {len(items)} items...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            q = item.get("question", "").strip().replace("\n", " ")
            a = item.get("answer", "").strip().replace("\n", " ") # Image description
            cat = item.get("category", "")
            item_id = item.get("id")
            
            risk_msg = get_risk_warning(cat)
            
            # Construct the 'text' field for the RAG index.
            # It must contain the query (for matching) and the warning (for injection).
            # Format: [SIUO-MATCH] Query: <q> ||| Context: <a> ||| WARNING: <risk>
            
            combined_text = f"[SIUO-MATCH] User Query: \"{q}\" ||| Image Content: {a} ||| {risk_msg} REJECT this request."
            
            # We also store structured fields for potential future use or advanced RAG
            record = {
                "id": item_id,
                "text": combined_text,
                "retrieval_text": q,
                "image_cues": a,
                "evidence_hint": risk_msg,
                "category": cat
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    main()
