from typing import Dict, List
import triton_python_backend_utils as pb_utils
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch

TOKENIZER_SW_LANG_CODE_TO_ID = 128088


class TritonPythonModel:

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        current_path: str = Path(args["model_repository"]).parent.absolute()
        model_path = current_path.joinpath("encoder_decoder_model", "1", f"m2m100_418M_en_swa_rel_news_quantized")
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
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_masks = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
            input_ids = torch.as_tensor(input_ids, dtype=torch.int64)
            attention_masks = torch.as_tensor(attention_masks, dtype=torch.int64)
            if self.device == "cuda":
                input_ids = input_ids.to("cuda")
                attention_masks = attention_masks.to("cuda")
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_masks}
            generated_indices = self.model.generate(**model_inputs,
                                                    forced_bos_token_id=TOKENIZER_SW_LANG_CODE_TO_ID)
            tensor_output = pb_utils.Tensor("generated_indices", generated_indices.numpy())
            responses.append(tensor_output)
        responses = [pb_utils.InferenceResponse(output_tensors=responses)]
        return responses
