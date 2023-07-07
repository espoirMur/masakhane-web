from typing import Dict, List
import numpy as np
import triton_python_backend_utils as pb_utils
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoConfig


TOKENIZER_SW_LANG_CODE_TO_ID = 128088


class TritonPythonModel:

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        current_path: str = Path(__file__).parent
        model_path = current_path.joinpath(f"m2m100_418M_en_swa_rel_news_quantized")
        print(f"Loading model from  the current path is {current_path} and the model path {model_path}", model_path.exists())
        self.device = "cpu" if args["model_instance_kind"] == "CPU" else "cuda"
        # more variables in https://github.com/triton-inference-server/python_backend/blob/main/src/python.cc
        self.model = ORTModelForSeq2SeqLM.from_pretrained(model_path,
                                                          decoder_file_name="decoder_model_quantized.onnx",
                                                          encoder_file_name="encoder_model_quantized.onnx")
        if self.device == "cuda":
            self.model = self.model.cuda()

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().tolist()]
            input_ids = input_ids.type(dtype=np.int32)
            if self.device == "cuda":
                input_ids = input_ids.to("cuda")
            generated_indices = self.model.generate(input_ids, forced_bos_token_id=TOKENIZER_SW_LANG_CODE_TO_ID)
            responses.append(pb_utils.InferenceResponse(generated_indices))
        return responses
