- should not put the model in a docker container, use the file storage instead and make it available as a volume to the container
- use a model registry to store models, build one with mlflow.
- Run different services for each model, and use a load balancer to route the requests to the right model.



torch-model-archiver --model-name MasaknaneEnSwaRelNews \
--version 1.0 \
--serialized-file src/torchserve/transformer_models/masakhane/m2m100_418M_en_swa_rel_news/pytorch_model.bin \
--handler src/torchserve/transformer_handler.py \
--extra-files "src/torchserve/transformer_models/masakhane/m2m100_418M_en_swa_rel_news/config.json,
               src/torchserve/transformer_models/masakhane/m2m100_418M_en_swa_rel_news/special_tokens_map.json,
               src/torchserve/transformer_models/masakhane/m2m100_418M_en_swa_rel_news/tokenizer_config.json,
               src/torchserve/transformer_models/masakhane/m2m100_418M_en_swa_rel_news/vocab.json,
               src/torchserve/transformer_models/masakhane/m2m100_418M_en_swa_rel_news/generation_config.json,
               src/torchserve/transformer_models/masakhane/m2m100_418M_en_swa_rel_news/sentencepiece.bpe.model"   



-- command to run the triton server:

docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 1024m \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:22.07-py3 \
  bash -c "pip install transformers==4.26.1 torch==1.13.1 && \
  tritonserver --model-repository=/models"



docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/triton_model_repository:/models/ nvcr.io/nvidia/tritonserver:22.07-py3  bash -c "pip install transformers==4.26.1 torch==1.13.1 && \
  tritonserver --model-repository=/models"


The key was to add `disable-auto-complete-config` config when run the model to enable the use of the custom config file, otherwise the server will try to generate a default configuration file which will not work.
