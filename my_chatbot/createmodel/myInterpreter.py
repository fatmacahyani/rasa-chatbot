from flask import Flask, request, jsonify
import re
import os

app = Flask(__name__)

# Initialize NER pipeline
ner_pipeline = None

def initialize_ner():
    """Initialize NER pipeline dengan multiple fallback options."""
    global ner_pipeline
    
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-keras"])
        
        # load pipeline
        from transformers import pipeline
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        print(" NER pipeline loaded successfully")
        return True
        
    except Exception as e:
        print(f" Error loading NER pipeline: {e}")
        print("  Using regex-only entity recognition")
        return False

def hybrid_entity_recognition(text):
    """Entity recognition dengan fallback ke regex jika BERT NER gagal."""
    entities = []
    
    # 1. BERT-based NER
    if ner_pipeline:
        try:
            bert_entities = ner_pipeline(text)
            for ent in bert_entities:
                entities.append({
                    "entity": ent["entity_group"],
                    "word": ent["word"],
                    "score": ent["score"],
                    "start": ent["start"],
                    "end": ent["end"],
                })
        except Exception as e:
            print(f"BERT NER error: {e}")

    # 2. Regex-based NER
    regex_patterns = {
        "jenjang": r"\b(?:magister|doktor|s2|s3)\b",
        "fakultas": r"\b(?:ELECTICS|INDSYS|MARTECH|CIVPLAN|SCIENTICS|SIMT|CREABIZ)\b",
        "fakultas_full": r"(?:teknologi elektro dan informatika cerdas|teknologi industri dan rekayasa sistem|teknik sipil[,\s]*perencanaan[,\s]*dan kebumian|teknologi kelautan|sains dan analitika data|sekolah interdisiplin manajemen dan teknologi|fakultas desain kreatif dan bisnis digital)",
        "prodi": r"(?:teknik informatika|teknik elektro|teknik sipil|statistika|manajemen teknologi|arsitektur|teknik kelautan|teknik material|teknik mesin|teknik kimia|teknik fisika|teknik industri|sistem informasi|matematika|fisika|kimia|biologi)",
    }

    for label, pattern in regex_patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Cek apakah sudah ada dari BERT
            overlap = False
            for existing in entities:
                if (match.start() >= existing["start"] and match.end() <= existing["end"]) or \
                   (existing["start"] >= match.start() and existing["end"] <= match.end()):
                    overlap = True
                    break
            
            if not overlap:
                entities.append({
                    "entity": label,
                    "word": match.group(),
                    "score": 1.0,
                    "start": match.start(),
                    "end": match.end(),
                })

    return entities


from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle

# Global variables untuk model
model = None
tokenizer = None
label_encoder = None

def load_intent_model(model_path="pasca_intent_model"):
    """Load model, tokenizer, dan label encoder."""
    global model, tokenizer, label_encoder
    
    try:
        if not os.path.exists(model_path):
            print(f" Model path tidak ditemukan: {model_path}")
            return False
            
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        label_encoder_path = f"{model_path}/label_encoder.pkl"
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
        else:
            print(f" Label encoder tidak ditemukan: {label_encoder_path}")
            return False
            
        model.eval()
        print(" Intent model loaded successfully")
        return True
        
    except Exception as e:
        print(f" Error loading intent model: {e}")
        return False

def get_intent(text, entity_list, modtext):
    """Prediksi intent dari input text."""
    global model, tokenizer, label_encoder
    
    if not all([model, tokenizer, label_encoder]):
        return {
            "intent": {"name": "nlu_fallback", "confidence": 0.1},
            "entities": entity_list,
            "text": text,
            "modtext": modtext
        }
    
    try:
        inputs = tokenizer(modtext, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=1).item()
            intent_name = label_encoder.inverse_transform([pred_idx])[0]
            confidence = probs[0][pred_idx].item()

        if text == '/session_start':
            intent_name = 'session_start'
            confidence = 1.0

        return {
            "intent": {
                "name": intent_name,
                "confidence": float(confidence)
            },
            "entities": entity_list,
            "text": text,
            "modtext": modtext
        }
        
    except Exception as e:
        print(f" Error in intent prediction: {e}")
        return {
            "intent": {"name": "nlu_fallback", "confidence": 0.1},
            "entities": entity_list,
            "text": text,
            "modtext": modtext
        }

def replace_entity(text, entity_value, placeholder):
    """Replace entity value dengan placeholder dalam text."""
    pattern = re.escape(entity_value)
    return re.sub(pattern, placeholder, text, flags=re.IGNORECASE)

def get_intent_and_entity(text):
    """Main function untuk mendapatkan intent dan entities."""
    entities = hybrid_entity_recognition(text)
    entity_list = []
    modtext = text
    
    for ent in entities:
        entity_type = ent['entity']
        
        # Mapping entity types sesuai dengan domain.yml
        if entity_type in ['jenjang']:
            entity_typex = 'jenjang'
        elif entity_type in ['fakultas', 'fakultas_full']:
            entity_typex = 'fakultas'
        elif entity_type in ['prodi']:
            entity_typex = 'prodi'
        elif entity_type == 'ORG':  # dari BERT NER
            entity_typex = 'program'
        else:
            entity_typex = entity_type
        
        my_entity = {
            'value': ent['word'], 
            'entity': entity_typex,
            'start': ent['start'],
            'end': ent['end'],
            'confidence': ent['score']
        }
        entity_list.append(my_entity)
        
        # Replace dengan placeholder untuk intent prediction
        if entity_type in ['jenjang']:
            modtext = replace_entity(modtext, ent['word'], "<jenjang>")
        elif entity_type in ['fakultas', 'fakultas_full']:
            modtext = replace_entity(modtext, ent['word'], "<fakultas>")
        elif entity_type in ['prodi']:
            modtext = replace_entity(modtext, ent['word'], "<prodi>")
        elif entity_type == 'ORG':
            modtext = replace_entity(modtext, ent['word'], "<program>")
            
    hasil = get_intent(text, entity_list, modtext)
    return hasil 

@app.route("/model/parse", methods=["POST"])
def parse():
    """Endpoint untuk parsing text seperti Rasa NLU."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_input = data.get("text", "")
        if not user_input:
            return jsonify({"error": "No text provided"}), 400
        
        print(f" Input: {user_input}")

        # Get intent and entities
        response = get_intent_and_entity(user_input)

        print(f" Response: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f" Error in /model/parse: {e}")
        return jsonify({
            "intent": {"name": "nlu_fallback", "confidence": 0.1},
            "entities": [],
            "text": user_input if 'user_input' in locals() else "",
            "error": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model_status = "loaded" if all([model, tokenizer, label_encoder]) else "not loaded"
    ner_status = "loaded" if ner_pipeline else "regex-only"
    
    return jsonify({
        "status": "OK",
        "message": "Pascasarjana ITS Intent Classifier",
        "intent_model": model_status,
        "ner_model": ner_status,
        "timestamp": "2025-10-20 07:48:24"
    })

@app.route("/", methods=["GET"])
def index():
    """Root endpoint."""
    return jsonify({
        "service": "Pascasarjana ITS Intent Classifier",
        "version": "1.0",
        "endpoints": {
            "parse": "/model/parse",
            "health": "/health"
        }
    })

if __name__ == "__main__":
    print("🚀 Starting Pascasarjana ITS Intent Classifier...")
    
    # Initialize NER pipeline
    initialize_ner()
    
    # Load intent model saat startup
    model_loaded = load_intent_model("pasca_intent_model")
    if not model_loaded:
        print(" Warning: Intent model not loaded, using fallback")
    
    # Start Flask app
    app.run(host="127.0.0.1", port=8000, debug=True)