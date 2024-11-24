
import os
import torch
import mlflow
from data.potter import HarryPotterDataset

# Set the tracking URI to your local ml-runs folder
mlflow.set_tracking_uri("file:///Users/I756185/Projekte/LiquidxLSTM/ml-runs/")

# Replace with your actual run ID
run_id = "75d2b9f2e64b48d4ab662db42541286c"
model_uri = f"runs:/{run_id}/model"

# Load the model
loaded_model = mlflow.pytorch.load_model(model_uri)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# hyper parameters
seq_length = 200
max_epochs = 100

file_path = os.path.join(os.getcwd(), "experimente/data/harrypotter_one.txt")
tokenizer_path = os.path.join(os.getcwd(), "experimente/data/bpe_tokenizer.json")
dataset = HarryPotterDataset(file_path, seq_length, device, tokenizer_path)

text = "Harry Potter was a wizard who lived in a castle."
encoded = dataset.encode(text)
print(encoded)

new_logits = encoded
for i in range(20):
    logits = loaded_model.forward(torch.Tensor([new_logits[-100:]]).to(dtype=torch.long))
    logits = torch.argmax(logits[:, -1, :], dim=1)
    new_logits.append(logits.tolist()[0])

decoded_text = dataset.decode(new_logits)
print(decoded_text.replace("Ġ", ""))
