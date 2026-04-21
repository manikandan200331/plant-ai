import requests

def predict_plant(image):
    try:
        response = requests.post(
            "https://plant-ai-model.hf.space/run/predict",
            files={"data": image}
        )

        result = response.json()

        # HuggingFace returns list
        data = result["data"]

        return {
            "disease": data[0],
            "confidence": data[1],
            "affected_percent": data[2],
            "reason": data[3],
            "organic_remedy": data[4],
            "chemical_remedy": data[5]
        }

    except Exception as e:
        return {"error": str(e)}