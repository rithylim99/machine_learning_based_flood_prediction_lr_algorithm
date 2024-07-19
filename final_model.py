import joblib
import numpy as np
model = joblib.load('trained_flood.joblib')


# Function to make a prediction and return the probability
def predict_flood_probability(water_level):
    # Prepare the input data as a 2D array
    input_data = np.array([[water_level]])

    # Make the prediction probability
    probabilities = model.predict_proba(input_data)

    # The probability of the positive class (flood) is the second column
    flood_probability = probabilities[0][1]

    return flood_probability


# Example usage
water_level = 5  # meters

flood_probability = predict_flood_probability(water_level)
print(f'When Water Level =5m then, Probability of Flood: {flood_probability * 100:.2f}%')