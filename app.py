#--------------------------------------------------------
# FAST API
import pickle
import uvicorn
from fastapi import FastAPI, Request
import pandas as pd
import lightgbm
model = pickle.load(open("model.pkl", "rb"))
print(model)
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Cars Recommender ML API'}

@app.post('/predict')
async def predict(request: Request):
    result = await request.json()
    df = pd.DataFrame.from_dict(result)

    # Use the loaded model to make predictions on the DataFrame
    prediction = model.predict(df)
    print(prediction)
    # Return the predictions as a JSON response
    return {"prediction": list(prediction)}