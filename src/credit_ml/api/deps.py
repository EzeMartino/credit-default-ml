from functools import lru_cache

from credit_ml.modeling.artifacts import load_pipeline, load_metadata

@lru_cache(maxsize=1)
def get_pipeline():
    return load_pipeline()

@lru_cache(maxsize=1)
def get_metadata():
    return load_metadata()