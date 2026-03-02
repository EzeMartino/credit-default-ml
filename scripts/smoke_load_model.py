from credit_ml.modeling.artifacts import load_pipeline, load_metadata

if __name__ == "__main__":
    pipe = load_pipeline()
    meta = load_metadata()
    print("Loaded pipeline:", type(pipe))
    print("Metadata keys:", list(meta.keys()))