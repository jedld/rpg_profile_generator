from model import Generator
import torch
import onnx

Z_DIM = 100
gen = Generator(Z_DIM)
gen.load_state_dict(torch.load(f"best_generator_model.pth"))
gen.eval()

dummy_input = torch.randn(1, Z_DIM)

output_onnx_path = "generator.onnx"

# Export the model
torch.onnx.export(gen,               # model being run
                  dummy_input,         # model input (or a tuple for multiple inputs)
                  output_onnx_path,    # where to save the model (can be a file or file-like object)
                  verbose=True)  



# Load the ONNX model
model_onnx = onnx.load(output_onnx_path)

# Verify the model has a valid schema
onnx.checker.check_model(model_onnx)
print("ONNX model is valid!")