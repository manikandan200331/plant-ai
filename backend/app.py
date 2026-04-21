# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory
from predict_disease import predict_plant
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
# ---------------- Predefined Q&A Data ----------------
qa_data = {
    "tomato_early_blight": {
        "en": {"answer": "Tomato Early Blight is a fungal disease caused by Alternaria solani. It appears as dark brown spots with yellow rings on older leaves, similar to a target or bullseye pattern. The disease spreads rapidly in warm and humid conditions, especially when temperatures are between 24-29°C. It weakens the plant slowly and reduces fruit production. To control it: remove and destroy infected leaves immediately, avoid overhead watering, ensure proper spacing between plants for good air circulation, spray Neem oil every week as organic remedy, and apply Mancozeb or Chlorothalonil fungicide for chemical control."},
        "ta": {"answer": "தக்காளி ஆரம்ப பிளைட் என்பது Alternaria solani என்ற பூஞ்சையால் வரும் நோய். இது பழைய இலைகளில் மஞ்சள் வளையங்களுடன் கூடிய கருப்பு-பழுப்பு புள்ளிகளாக தோன்றும். 24-29°C வெப்பநிலையில் ஈரமான சூழலில் வேகமாக பரவும். இது செடியை மெதுவாக பலவீனப்படுத்தி காய்கள் உற்பத்தியை குறைக்கும். தீர்வுகள்: பாதிக்கப்பட்ட இலைகளை உடனே அகற்றவும், மேலே நீர் ஊற்றுவதை தவிர்க்கவும், செடிகளுக்கு இடையே நல்ல இடைவெளி வைக்கவும், வேப்ப எண்ணெய் தெளிக்கவும், மற்றும் Mancozeb மருந்து பயன்படுத்தவும்."},
        "tg": {"answer": "Tomato Early Blight fungal disease da, Alternaria solani cause pannuthu. Patha leaves la dark brown spots yellow rings sathe varum, bullseye pattern maari irukkum. 24-29°C temperature la humid weather la fast-a spread aagum. Plant-a slowly weak pannum, fruit production kurayum. Solution: infected leaves immediately remove pannu, overhead watering avoid pannu, plants-ku proper spacing vaikka, weekly Neem oil spray pannu, chemical-a Mancozeb or Chlorothalonil use pannu."}
    },
    "tomato_late_blight": {
        "en": {"answer": "Tomato Late Blight is a serious disease caused by Phytophthora infestans. It appears as water-soaked, dark green to brown patches on leaves and stems, with a white mold visible on the underside of leaves in humid conditions. It spreads extremely fast in cool and wet weather between 10-20°C and can destroy an entire crop within days. To control it: remove and burn infected plants immediately, avoid wetting the foliage, improve air circulation, apply copper-based fungicide every 7-10 days, and use Mancozeb or Metalaxyl for better chemical control."},
        "ta": {"answer": "தக்காளி லேட் பிளைட் Phytophthora infestans என்ற நுண்ணுயிரியால் வரும் கடுமையான நோய். இலைகளில் நீர் நனைந்த மாதிரி கருப்பு-பழுப்பு திட்டுகள் தோன்றும், ஈரமான சூழலில் இலையின் அடிப்பகுதியில் வெள்ளை பூஞ்சை தெரியும். 10-20°C குளிர் மற்றும் மழை காலத்தில் மிக வேகமாக பரவி பயிர் முழுவதையும் சில நாட்களில் அழிக்கும். தீர்வுகள்: பாதிக்கப்பட்ட செடிகளை உடனே அகற்றி எரிக்கவும், இலைகளை நனைக்காதீர்கள், காற்றோட்டம் மேம்படுத்தவும், காப்பர் பூஞ்சை மருந்து 7-10 நாட்களுக்கு ஒரு முறை தெளிக்கவும்."},
        "tg": {"answer": "Tomato Late Blight romba serious disease da, Phytophthora infestans cause pannuthu. Leaves la water-soaked dark patches varum, humid-a irundha leaf bottom la white mold teriyum. 10-20°C cool wet weather la super fast spread aagum, entire crop-a few days la destroy pannidum. Solution: infected plants immediately remove panni burn pannu, leaves wet aagaama paakku, air circulation improve pannu, copper fungicide every 7-10 days spray pannu, Mancozeb or Metalaxyl use pannu."}
    },
    "potato_early_blight": {
        "en": {"answer": "Potato Early Blight is caused by the fungus Alternaria solani. It first appears on older, lower leaves as small brown spots that gradually enlarge with concentric rings forming a target-like pattern. The disease thrives in warm humid weather and spreads through wind and water splash. Infected leaves turn yellow and drop early, reducing the plant's ability to produce food through photosynthesis. To manage it: remove infected leaves early, avoid overhead irrigation, maintain proper plant spacing, apply Neem oil spray weekly, and use Mancozeb fungicide for chemical treatment."},
        "ta": {"answer": "உருளைக்கிழங்கு ஆரம்ப பிளைட் Alternaria solani பூஞ்சையால் வருகிறது. முதலில் கீழ் இலைகளில் சிறிய பழுப்பு புள்ளிகளாக தோன்றி, வளையங்களுடன் பெரிதாகும். வெப்பமான ஈரமான சூழலில் காற்று மற்றும் நீர் மூலம் பரவும். பாதிக்கப்பட்ட இலைகள் மஞ்சளாகி உதிரும், இதனால் ஒளிச்சேர்க்கை குறைந்து செடி பலவீனமாகும். தீர்வு: ஆரம்பத்திலேயே பாதிக்கப்பட்ட இலைகளை அகற்றவும், மேல்நோக்கிய நீர்ப்பாசனம் தவிர்க்கவும், வேப்ப எண்ணெய் தெளிக்கவும், Mancozeb பூஞ்சை மருந்து பயன்படுத்தவும்."},
        "tg": {"answer": "Potato Early Blight Alternaria solani fungus cause pannuthu. First-a older leaves la small brown spots varum, slowly target pattern maari rings sathe periyathaagum. Warm humid weather la wind and water through spread aagum. Infected leaves yellow aagi fall aagum, photosynthesis kurainju plant weak aagum. Solution: early-a infected leaves remove pannu, overhead watering avoid pannu, plant spacing sariya vaikka, weekly Neem oil spray pannu, Mancozeb fungicide use pannu."}
    },
    "potato_late_blight": {
        "en": {"answer": "Potato Late Blight is one of the most destructive plant diseases, caused by Phytophthora infestans. It appears as dark water-soaked lesions on leaves and stems, with white fungal growth on the underside of leaves. The tubers underground also get infected, turning brown and rotting. It spreads rapidly in cool and wet conditions, especially during monsoon season. To control it: remove and destroy infected plants completely, avoid excess watering, provide good drainage, spray copper fungicide every 7 days, and use Metalaxyl or Cymoxanil for effective chemical control."},
        "ta": {"answer": "உருளைக்கிழங்கு லேட் பிளைட் Phytophthora infestans காரணமாக வரும் மிகவும் அழிவுகரமான நோய். இலைகள் மற்றும் தண்டுகளில் கருமையான நீர் நனைந்த புண்கள் தோன்றும், இலையின் அடியில் வெள்ளை பூஞ்சை வளரும். நிலத்தடி கிழங்குகளும் பாதிக்கப்பட்டு அழுகும். மழை மற்றும் குளிர் காலத்தில் மிக வேகமாக பரவும். தீர்வு: பாதிக்கப்பட்ட செடிகளை முற்றிலும் அகற்றி அழிக்கவும், அதிக நீர் தவிர்க்கவும், நல்ல வடிகால் வசதி ஏற்படுத்தவும், 7 நாட்களுக்கு ஒரு முறை காப்பர் பூஞ்சை மருந்து தெளிக்கவும்."},
        "tg": {"answer": "Potato Late Blight romba dangerous disease da, Phytophthora infestans cause pannuthu. Leaves and stems la dark water-soaked patches varum, leaf bottom la white fungal growth teriyum. Underground tubers um infected aagi rot aagum. Monsoon season la cool wet weather la super fast spread aagum. Solution: infected plants completely remove panni destroy pannu, excess watering avoid pannu, good drainage vaikka, every 7 days copper fungicide spray pannu, Metalaxyl or Cymoxanil use pannu."}
    },
    "pepper_bell_bacterial_spot": {
        "en": {"answer": "Pepper Bell Bacterial Spot is caused by Xanthomonas bacteria. It appears as small, water-soaked spots on leaves that turn brown or black with yellow halos around them. The spots may also appear on fruits, causing raised, scab-like lesions. The bacteria spread through rain, irrigation water, contaminated tools, and infected seeds. It thrives in warm and wet weather above 25°C. To control it: remove and destroy infected leaves and fruits immediately, avoid overhead watering, disinfect garden tools regularly, apply copper-based bactericide every 7-10 days, and use disease-free certified seeds for future planting."},
        "ta": {"answer": "மிளகாய் பாக்டீரியல் ஸ்பாட் Xanthomonas பாக்டீரியாவால் வருகிறது. இலைகளில் நீர் நனைந்த சிறிய புள்ளிகளாக தோன்றி மஞ்சள் வளையங்களுடன் கருப்பாகும். காய்களிலும் புண்கள் தோன்றும். மழை நீர், பாசன நீர், மாசுபட்ட கருவிகள் மற்றும் நோய்வாய்பட்ட விதைகள் மூலம் பரவும். 25°C-க்கு மேல் வெப்பமான ஈரமான சூழலில் அதிகமாக பரவும். தீர்வு: பாதிக்கப்பட்ட இலைகள் மற்றும் காய்களை உடனே அகற்றவும், மேல்நோக்கிய நீர்ப்பாசனம் தவிர்க்கவும், தோட்ட கருவிகளை கிருமி நாசினியால் சுத்தம் செய்யவும், காப்பர் பாக்டீரிசைட் 7-10 நாட்களுக்கு ஒரு முறை தெளிக்கவும்."},
        "tg": {"answer": "Pepper Bell Bacterial Spot Xanthomonas bacteria cause pannuthu. Leaves la water-soaked small spots varum, yellow halo sathe brown or black aagum. Fruits la um scab-like lesions varum. Rain water, irrigation water, contaminated tools and infected seeds through spread aagum. 25°C above warm wet weather la fast-a spread aagum. Solution: infected leaves and fruits immediately remove pannu, overhead watering avoid pannu, garden tools regularly disinfect pannu, every 7-10 days copper bactericide spray pannu, future-ku disease-free certified seeds use pannu."}
    }
}
# ---------------- Q&A Route ----------------
@app.route("/get_answer", methods=["POST"])
def get_answer():
    try:
        data = request.json
        qid = data.get("qid")        # eg: tomato_early_blight
        lang = data.get("lang") 
        answer = "answer not available"     # en / ta / tg
        if qid in qa_data and lang in qa_data[qid]:
            answer = qa_data[qid][lang]["answer"]
        else:
            answer = "Answer not available"
        return jsonify({"answer": answer})
    except Exception as e:
        print("Error in /get_answer:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- Predict Route ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Save temp file
        file_path = os.path.join(os.getcwd(), "temp.jpg")
        file.save(file_path)

        # Call your AI predict function
        disease, confidence, affected, reason, organic, chemical = predict_plant(file_path)

        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "affected": affected,
            "reason": reason,
            "organic": organic,
            "chemical": chemical
        })
    except Exception as e:
        print("Error in /predict:", e)
        return jsonify({"error": str(e)}), 500
    
#index route   
@app.route("/")
def index():
    return send_from_directory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend"),
        "index.html"
    )

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Server starting...")
    port = int(os.environ.get("PORT", 5000))  # Render automatically sets PORT
    app.run(host="0.0.0.0", port=port)