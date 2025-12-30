import sys, os
import tensorflow as tf
import numpy as np
import time

from keras import ops
from keras.layers import Lambda

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from config import get_config, summarize_config
from mode_config import get_mode_config

from training.trainer import train_embeddings


def main():

    start_time = time.time()

    config = get_config()
    # summarize_config(config)
    mode_config = get_mode_config(config["mode"])
    summarize_config(mode_config)


    model = train_embeddings(config=config, mode_config=mode_config)
    
    end_time = time.time()
    print("Training time: ", end_time - start_time)

    # # --- Evaluate after training ---
    print("\n🔍 Running post-training evaluation...")
    validation_config = config.copy()
    validation_config["is_validation"] = True
    gen_cls = mode_config["generator"]
    gen_kwargs = mode_config.get("generator_kwargs", {})
    eval_gen = gen_cls(validation_config, **gen_kwargs)

    # output_adapter = mode_config.get("output_adapter", {})
    # adapter = build_output_adapter(output_adapter)
    # eval_gen = AdaptedGenerator(eval_gen, adapter)

    eval_gen.preload_validation_data()   
    evaluate_cls = mode_config["evaluation_function"]
    eval_kwargs = mode_config.get("evaluation_function_kwargs", {})
    evaluate_cls(model, iter(eval_gen), num_examples=50, **eval_kwargs)
      
if __name__ == "__main__":
    main()
