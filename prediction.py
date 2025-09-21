
sample = [[5.3, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
print("Predicted species:", iris.target_names[prediction][0])
