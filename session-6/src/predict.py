from typing import List

import torch

from model import RegressionModel

@torch.no_grad()
def predict(input_features: List[float]):
    # load the checkpoint from the correct path
    checkpoint = torch.load("/checkpoints/checkpoint.pt")  # Adjust the path as needed

    # Instantiate the model and load the state dict
    model = RegressionModel(checkpoint["input_size"], checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Input features is a list of floats. We have to convert it to a tensor of the correct shape
    x = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)

    # Now we have to do the same normalization we did when training
    x = (x - checkpoint["x_mean"]) / checkpoint["x_std"]

    # We get the output of the model and we print it
    output = model(x)

    # We have to revert the target normalization that we did when training
    output = output * checkpoint["y_std"] + checkpoint["y_mean"]

    print(f"The predicted price is: ${output.item()*1000:.2f}")

# Example usage:
# input_features = [0.24522,0,9.9,0,0.544,5.782,71.7,4.0317,4,304,18.4,396.9,15.94]  # Replace with your actual input features
# predict(input_features)
