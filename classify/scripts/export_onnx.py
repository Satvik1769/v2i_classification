import torch
import torch.onnx

# set image size, square image only
image_size = 180  # for (180, 180, 3) image
dummy_input = torch.randn(1, 3, image_size, image_size)  #


model = torch.load(
    "../models/vehicle_dataset/vehicle_test/best_val_loss.pt",
    map_location=torch.device("cpu"),
)
model = torch.nn.Sequential(model, torch.nn.Softmax())
model.eval()

torch.onnx.export(
    model,
    dummy_input,
    "../models/vehicle_dataset/vehicle_test/vehicle_test.onnx",
    verbose=True,
)
