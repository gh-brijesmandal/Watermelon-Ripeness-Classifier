import joblib

model = joblib.load("ripeness_model.joblib")

print(model)

ripe_data = [[0.175343,0.030400557,
         0.0147916666666667,
         -3.08924727068419,
         120,
         1.956615,
         3.800893,
         120
         ]]

#samples from ai
# Sample 1: Classic hard, under-ripe melon
unripe_sample_1 = [[0.93, 0.49, 0.066, 5.8, 255, 0.42, 3.8, 255]]

# Sample 2: Slightly under-ripe, starting to transition
unripe_sample_2 = [ [0.89, 0.46, 0.063, 5.4, 245, 0.35, 3.5, 245]]

# Sample 1: Perfect premium ripe melon (Clear, crisp ring)
ripe_sample_1 = [ [0.83, 0.43, 0.053, 8.1, 198, 0.13, 2.8, 198]]

# Sample 2: Great ripe melon, slightly larger size (lower pitch, but clean peak)
ripe_sample_2 = [[0.80, 0.41, 0.051, 8.5, 190, 0.09, 2.7, 190]]

# Sample 1: Classic mealy, soft overripe melon
overripe_sample_1 = [[0.58, 0.28, 0.038, 14.1, 105, -0.29, 2.1, 105]]

# Sample 2: Heavily degraded, hollow-heart or mushy melon
overripe_sample_2 = [ [0.62, 0.30, 0.039, 13.8, 110, -0.24, 2.2, 110]]


pipeline = model["pipeline"]
encoder = model["label_encoder"]
feature_names = model["feature_columns"]

print(f"Loaded {model["model_name"]} model.")
print(f"It expects {feature_names}\n")

#Prediction
numeric_prediction = pipeline.predict(ripe_sample_2)

text_label = encoder.inverse_transform(numeric_prediction)[0]

print(f"Predicted Numerical Class: {numeric_prediction[0]}")
print(f"Predicted Ripeness Label: {text_label.upper()}")
