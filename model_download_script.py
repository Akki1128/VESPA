from transformers import AutoTokenizer, AutoModel
import torch
import onnx

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

dummy_input = tokenizer("This is a sample sentence.", return_tensors="pt")

onnx_file_path = "minilm-l6-v2.onnx"
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    onnx_file_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
    opset_version=11
)

print(f"Model converted to ONNX format and saved as {onnx_file_path}")
