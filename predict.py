import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from train import CNN

# Path to your drawn digit
img_folder = "C:/Users/itsdi/mnist/test_image"
img_name   = None  # will be auto-detected below

# Preprocessing parameters
resize_dim = 28   # MNIST images are 28x28
mean, std = (0.1307,), (0.3081,)

def find_image():
    for fn in os.listdir(img_folder):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(img_folder, fn)
    raise FileNotFoundError(f"No .png/.jpg found in {img_folder}")

def main():
    global img_name
    img_name = find_image()
    # load and preprocess
    img = Image.open(img_name).convert("L")  # grayscale
    transform = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tensor = transform(img).unsqueeze(0)  # add batch dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        tensor = tensor.to(device)
        output = model(tensor)
        pred = output.argmax(dim=1, keepdim=True).item()
    print(f"Image: {img_name}  →  Predicted Digit: {pred}")

if __name__ == "__main__":
    main()
