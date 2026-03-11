from credit_ml.torch.inference import load_torch_artifacts

model, scaler, meta = load_torch_artifacts()

print(meta["model_type"])
print(meta["input_dim"])
print(meta["validation_roc_auc"])