"""
Inference pipeline: takes a query + retrieved context chunks
and generates an answer using the fine-tuned Mistral-7B model.
"""

from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from loguru import logger


class RAGGenerator:
    """
    Wraps the fine-tuned Mistral-7B for answer generation.
    Takes a query and a list of retrieved context chunks,
    formats them into a prompt, and decodes the response.
    """

    PROMPT_TEMPLATE = (
        "<s>[INST] Answer the following question based only on the provided context. "
        "If the context does not contain enough information, say so clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question} [/INST]"
    )

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_new_tokens = config["model"]["max_new_tokens"]
        self.temperature = config["model"]["temperature"]
        self.top_p = config["model"]["top_p"]
        self.device = config["model"]["device"]

        self._load_model()

    def _load_model(self):
        base_model = self.config["model"]["base_model"]
        finetuned_path = self.config["model"]["finetuned_model_path"]

        logger.info(f"Loading tokenizer from {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        logger.info(f"Loading base model: {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )

        logger.info(f"Loading LoRA adapter from {finetuned_path}")
        self.model = PeftModel.from_pretrained(base, finetuned_path)
        self.model.eval()
        logger.info("RAGGenerator model loaded and ready")

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Concatenate retrieved chunks into a context block."""
        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("metadata", {}).get("source", f"doc_{i}")
            parts.append(f"[{i+1}] (Source: {source})\n{chunk['text']}")
        return "\n\n".join(parts)

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer given query + retrieved chunks.
        Returns dict: {answer, prompt, num_context_chunks, sources}.
        """
        context = self._build_context(chunks)
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=query)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        sources = list({c.get("doc_id", "") for c in chunks})

        return {
            "answer": answer,
            "prompt": prompt,
            "num_context_chunks": len(chunks),
            "sources": sources,
        }
