import logging

import numpy as np
from transformers import AutoProcessor
from transformers import GemmaTokenizerFast

import openpi.shared.download as download

PALIGEMMA_EOS_TOKEN = 1


class TOASTTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        *,
        toast_tokenizer_path: str,
        sample_segmentation: bool = False,
        alpha: float = 0.1,
        nbest_size: int = 64,
    ):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        self._paligemma_tokenizer = GemmaTokenizerFast(vocab_file=str(path))

        # Instantiate TOAST tokenizer (a local directory or a Hugging Face repository)
        self._toast_tokenizer = AutoProcessor.from_pretrained(toast_tokenizer_path, trust_remote_code=True)
        # Whether (and how) the segmentation of the action tokens is sampled in tokenize().
        self._toast_tokenizer.sample = sample_segmentation
        self._toast_tokenizer.alpha = alpha
        self._toast_tokenizer.nbest_size = nbest_size

        self._skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens
        self._paligemma_vocab_size = self._paligemma_tokenizer.vocab_size

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_special_tokens=True)

        if actions is not None:
            # Tokenize actions with TOAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._toast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by TOAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False)
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_special_tokens=False)
                + [PALIGEMMA_EOS_TOKEN]
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

        # Extract actions from TOAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(
                decoded_tokens.split("Action: ")[1].split("|")[0].strip(), add_special_tokens=False
            )
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)

        return self._toast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)

        return self._paligemma_vocab_size - 1 - self._skip_tokens - tokens


