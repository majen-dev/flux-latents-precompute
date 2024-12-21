# base class for platform strategies. this file defines the interface for strategies

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import safe_open, save_file
from transformers import CLIPTokenizer

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import dataset_metadata_utils, utils


def get_compatible_dtypes(dtype: Optional[Union[str, torch.dtype]]) -> List[torch.dtype]:
    if dtype is None:
        # all dtypes are acceptable
        return get_available_dtypes()

    dtype = utils.str_to_dtype(dtype) if isinstance(dtype, str) else dtype
    compatible_dtypes = [torch.float32]
    if dtype.itemsize == 1:  # fp8
        compatible_dtypes.append(torch.bfloat16)
        compatible_dtypes.append(torch.float16)
    compatible_dtypes.append(dtype)  # add the specified: bf16, fp16, one of fp8
    return compatible_dtypes


def get_available_dtypes() -> List[torch.dtype]:
    """
    Returns the list of available dtypes for latents caching. Higher precision is preferred.
    """
    return [torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]


def remove_lower_precision_values(tensor_dict: Dict[str, torch.Tensor], keys_without_dtype: list[str]) -> None:
    """
    Removes lower precision values from tensor_dict.
    """
    available_dtypes = get_available_dtypes()
    available_dtype_suffixes = [f"_{utils.dtype_to_normalized_str(dtype)}" for dtype in available_dtypes]

    for key_without_dtype in keys_without_dtype:
        available_itemsize = None
        for dtype, dtype_suffix in zip(available_dtypes, available_dtype_suffixes):
            key = key_without_dtype + dtype_suffix

            if key in tensor_dict:
                if available_itemsize is None:
                    available_itemsize = dtype.itemsize
                elif available_itemsize > dtype.itemsize:
                    # if higher precision latents are already cached, remove lower precision latents
                    del tensor_dict[key]


def get_compatible_dtype_keys(
    dict_keys: set[str], keys_without_dtype: list[str], dtype: Optional[Union[str, torch.dtype]]
) -> list[Optional[str]]:
    """
    Returns the list of keys with the specified dtype or higher precision dtype. If the specified dtype is None, any dtype is acceptable.
    If the key is not found, it returns None.
    If the key in dict_keys doesn't have dtype suffix, it is acceptable, because it it long tensor.

    :param dict_keys: set of keys in the dictionary
    :param keys_without_dtype: list of keys without dtype suffix to check
    :param dtype: dtype to check, or None for any dtype
    :return: list of keys with the specified dtype or higher precision dtype. If the key is not found, it returns None for that key.
    """
    compatible_dtypes = get_compatible_dtypes(dtype)
    dtype_suffixes = [f"_{utils.dtype_to_normalized_str(dt)}" for dt in compatible_dtypes]

    available_keys = []
    for key_without_dtype in keys_without_dtype:
        available_key = None
        if key_without_dtype in dict_keys:
            available_key = key_without_dtype
        else:
            for dtype_suffix in dtype_suffixes:
                key = key_without_dtype + dtype_suffix
                if key in dict_keys:
                    available_key = key
                    break
        available_keys.append(available_key)

    return available_keys


class TokenizeStrategy:
    _strategy = None  # strategy instance: actual strategy class

    _re_attention = re.compile(
        r"""\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
        re.X,
    )

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TokenizeStrategy"]:
        return cls._strategy

    def _load_tokenizer(
        self,
        model_class: Any,
        model_id: str,
        subfolder: Optional[str] = None,
        tokenizer_cache_dir: Optional[str] = None,
    ) -> Any:
        tokenizer = None
        if tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(tokenizer_cache_dir, model_id.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = model_class.from_pretrained(local_tokenizer_path)  # same for v1 and v2

        if tokenizer is None:
            tokenizer = model_class.from_pretrained(model_id, subfolder=subfolder)

        if tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        return tokenizer

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        raise NotImplementedError

    def tokenize_with_weights(self, text: Union[str, List[str]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        returns: [tokens1, tokens2, ...], [weights1, weights2, ...]
        """
        raise NotImplementedError

    def _get_weighted_input_ids(
        self, tokenizer: CLIPTokenizer, text: str, max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        max_length includes starting and ending tokens.
        """

        def parse_prompt_attention(text):
            """
            Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
            Accepted tokens are:
            (abc) - increases attention to abc by a multiplier of 1.1
            (abc:3.12) - increases attention to abc by a multiplier of 3.12
            [abc] - decreases attention to abc by a multiplier of 1.1
            \( - literal character '('
            \[ - literal character '['
            \) - literal character ')'
            \] - literal character ']'
            \\ - literal character '\'
            anything else - just text
            >>> parse_prompt_attention('normal text')
            [['normal text', 1.0]]
            >>> parse_prompt_attention('an (important) word')
            [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
            >>> parse_prompt_attention('(unbalanced')
            [['unbalanced', 1.1]]
            >>> parse_prompt_attention('\(literal\]')
            [['(literal]', 1.0]]
            >>> parse_prompt_attention('(unnecessary)(parens)')
            [['unnecessaryparens', 1.1]]
            >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
            [['a ', 1.0],
            ['house', 1.5730000000000004],
            [' ', 1.1],
            ['on', 1.0],
            [' a ', 1.1],
            ['hill', 0.55],
            [', sun, ', 1.1],
            ['sky', 1.4641000000000006],
            ['.', 1.1]]
            """

            res = []
            round_brackets = []
            square_brackets = []

            round_bracket_multiplier = 1.1
            square_bracket_multiplier = 1 / 1.1

            def multiply_range(start_position, multiplier):
                for p in range(start_position, len(res)):
                    res[p][1] *= multiplier

            for m in TokenizeStrategy._re_attention.finditer(text):
                text = m.group(0)
                weight = m.group(1)

                if text.startswith("\\"):
                    res.append([text[1:], 1.0])
                elif text == "(":
                    round_brackets.append(len(res))
                elif text == "[":
                    square_brackets.append(len(res))
                elif weight is not None and len(round_brackets) > 0:
                    multiply_range(round_brackets.pop(), float(weight))
                elif text == ")" and len(round_brackets) > 0:
                    multiply_range(round_brackets.pop(), round_bracket_multiplier)
                elif text == "]" and len(square_brackets) > 0:
                    multiply_range(square_brackets.pop(), square_bracket_multiplier)
                else:
                    res.append([text, 1.0])

            for pos in round_brackets:
                multiply_range(pos, round_bracket_multiplier)

            for pos in square_brackets:
                multiply_range(pos, square_bracket_multiplier)

            if len(res) == 0:
                res = [["", 1.0]]

            # merge runs of identical weights
            i = 0
            while i + 1 < len(res):
                if res[i][1] == res[i + 1][1]:
                    res[i][0] += res[i + 1][0]
                    res.pop(i + 1)
                else:
                    i += 1

            return res

        def get_prompts_with_weights(text: str, max_length: int):
            r"""
            Tokenize a list of prompts and return its tokens with weights of each token. max_length does not include starting and ending token.

            No padding, starting or ending token is included.
            """
            truncated = False

            texts_and_weights = parse_prompt_attention(text)
            tokens = []
            weights = []
            for word, weight in texts_and_weights:
                # tokenize and discard the starting and the ending token
                token = tokenizer(word).input_ids[1:-1]
                tokens += token
                # copy the weight by length of token
                weights += [weight] * len(token)
                # stop if the text is too long (longer than truncation limit)
                if len(tokens) > max_length:
                    truncated = True
                    break
            # truncate
            if len(tokens) > max_length:
                truncated = True
                tokens = tokens[:max_length]
                weights = weights[:max_length]
            if truncated:
                logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
            return tokens, weights

        def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad):
            r"""
            Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
            """
            tokens = [bos] + tokens + [eos] + [pad] * (max_length - 2 - len(tokens))
            weights = [1.0] + weights + [1.0] * (max_length - 1 - len(weights))
            return tokens, weights

        if max_length is None:
            max_length = tokenizer.model_max_length

        tokens, weights = get_prompts_with_weights(text, max_length - 2)
        tokens, weights = pad_tokens_and_weights(
            tokens, weights, max_length, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        return torch.tensor(tokens).unsqueeze(0), torch.tensor(weights).unsqueeze(0)

    def _get_input_ids(
        self, tokenizer: CLIPTokenizer, text: str, max_length: Optional[int] = None, weighted: bool = False
    ) -> torch.Tensor:
        """
        for SD1.5/2.0/SDXL
        TODO support batch input
        """
        if max_length is None:
            max_length = tokenizer.model_max_length - 2

        if weighted:
            input_ids, weights = self._get_weighted_input_ids(tokenizer, text, max_length)
        else:
            input_ids = tokenizer(
                text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
            ).input_ids

        if max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
                # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
                for i in range(
                    1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                    # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77

            if weighted:
                weights = weights.squeeze(0)
                new_weights = torch.ones(input_ids.shape)
                for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    b = i // (tokenizer.model_max_length - 2)
                    new_weights[b, 1 : 1 + tokenizer.model_max_length - 2] = weights[
                        i : i + tokenizer.model_max_length - 2
                    ]
                weights = new_weights

        if weighted:
            return input_ids, weights
        return input_ids


class TextEncodingStrategy:
    _strategy = None  # strategy instance: actual strategy class

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TextEncodingStrategy"]:
        return cls._strategy

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Encode tokens into embeddings and outputs.
        :param tokens: list of token tensors for each TextModel
        :return: list of output embeddings for each architecture
        """
        raise NotImplementedError

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
        weights: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Encode tokens into embeddings and outputs.
        :param tokens: list of token tensors for each TextModel
        :param weights: list of weight tensors for each TextModel
        :return: list of output embeddings for each architecture
        """
        raise NotImplementedError


class TextEncoderOutputsCachingStrategy:
    _strategy = None  # strategy instance: actual strategy class

    def __init__(
        self,
        architecture: str,
        cache_to_disk: bool,
        batch_size: Optional[int],
        skip_disk_cache_validity_check: bool,
        max_token_length: int,
        masked: bool = False,
        is_partial: bool = False,
        is_weighted: bool = False,
    ) -> None:
        """
        max_token_length: maximum token length for the model. Including/excluding starting and ending tokens depends on the model.
        """
        self._architecture = architecture
        self._cache_to_disk = cache_to_disk
        self._batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check
        self._max_token_length = max_token_length
        self._masked = masked
        self._is_partial = is_partial
        self._is_weighted = is_weighted  # enable weighting by `()` or `[]` in the prompt

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TextEncoderOutputsCachingStrategy"]:
        return cls._strategy

    @property
    def architecture(self):
        return self._architecture

    @property
    def max_token_length(self):
        return self._max_token_length

    @property
    def masked(self):
        return self._masked

    @property
    def cache_to_disk(self):
        return self._cache_to_disk

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def cache_suffix(self):
        suffix_masked = "_m" if self.masked else ""
        return f"_{self.architecture.lower()}_{self.max_token_length}{suffix_masked}_te.safetensors"

    @property
    def is_partial(self):
        return self._is_partial

    @property
    def is_weighted(self):
        return self._is_weighted

    def get_cache_path(self, absolute_path: str) -> str:
        return os.path.splitext(absolute_path)[0] + self.cache_suffix

    def load_from_disk(self, cache_path: str, caption_index: int) -> list[Optional[torch.Tensor]]:
        raise NotImplementedError

    def load_from_disk_for_keys(
        self, cache_path: str, caption_index: int, base_keys: list[str]
    ) -> list[Optional[torch.Tensor]]:
        """
        get tensors for keys_without_dtype, without dtype suffix. if the key is not found, it returns None.
        all dtype tensors are returned, because cache validation is done in advance.
        """
        with safe_open(cache_path, framework="pt") as f:
            metadata = f.metadata()
            version = metadata.get("format_version", "0.0.0")
            major, minor, patch = map(int, version.split("."))
            if major > 1:  # or (major == 1 and minor > 0):
                if not self.load_version_warning_printed:
                    self.load_version_warning_printed = True
                    logger.warning(
                        f"Existing latents cache file has a higher version {version} for {cache_path}. This may cause issues."
                    )

            dict_keys = f.keys()
            results = []
            compatible_keys = self.get_compatible_output_keys(dict_keys, caption_index, base_keys, None)
            for key in compatible_keys:
                results.append(f.get_tensor(key) if key is not None else None)

        return results

    def is_disk_cached_outputs_expected(
        self, cache_path: str, prompts: list[str], preferred_dtype: Optional[Union[str, torch.dtype]]
    ) -> bool:
        raise NotImplementedError

    def get_key_suffix(self, prompt_id: int, dtype: Optional[Union[str, torch.dtype]] = None) -> str:
        """
        masked: may be False even if self.masked is True. It is False for some outputs.
        """
        key_suffix = f"_{prompt_id}"
        if dtype is not None and dtype.is_floating_point:  # float tensor only
            key_suffix += "_" + utils.dtype_to_normalized_str(dtype)
        return key_suffix

    def get_compatible_output_keys(
        self, dict_keys: set[str], caption_index: int, base_keys: list[str], dtype: Optional[Union[str, torch.dtype]]
    ) -> list[Optional[str], Optional[str]]:
        """
        returns the list of keys with the specified dtype or higher precision dtype. If the specified dtype is None, any dtype is acceptable.
        """
        key_suffix = self.get_key_suffix(caption_index, None)
        keys_without_dtype = [k + key_suffix for k in base_keys]
        return get_compatible_dtype_keys(dict_keys, keys_without_dtype, dtype)

    def _default_is_disk_cached_outputs_expected(
        self,
        cache_path: str,
        captions: list[str],
        base_keys: list[tuple[str, bool]],
        preferred_dtype: Optional[Union[str, torch.dtype]],
    ):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(cache_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            with utils.MemoryEfficientSafeOpen(cache_path) as f:
                keys = f.keys()
                metadata = f.metadata()

            # check captions in metadata
            for i, caption in enumerate(captions):
                if metadata.get(f"caption{i+1}") != caption:
                    return False

                compatible_keys = self.get_compatible_output_keys(keys, i, base_keys, preferred_dtype)
                if any(key is None for key in compatible_keys):
                    return False
        except Exception as e:
            logger.error(f"Error loading file: {cache_path}")
            raise e

        return True

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: list[Any],
        text_encoding_strategy: TextEncodingStrategy,
        batch: list[tuple[utils.ImageInfo, int, str]],
    ):
        raise NotImplementedError

    def save_outputs_to_disk(
        self, cache_path: str, caption_index: int, caption: str, keys: list[str], outputs: list[torch.Tensor]
    ):
        tensor_dict = {}

        overwrite = False
        if os.path.exists(cache_path):
            # load existing safetensors and update it
            overwrite = True

            with utils.MemoryEfficientSafeOpen(cache_path) as f:
                metadata = f.metadata()
                keys = f.keys()
                for key in keys:
                    tensor_dict[key] = f.get_tensor(key)
            assert metadata["architecture"] == self.architecture

            file_version = metadata.get("format_version", "0.0.0")
            major, minor, patch = map(int, file_version.split("."))
            if major > 1 or (major == 1 and minor > 0):
                self.save_version_warning_printed = True
                logger.warning(
                    f"Existing latents cache file has a higher version {file_version} for {cache_path}. This may cause issues."
                )
        else:
            metadata = {}
            metadata["architecture"] = self.architecture
            metadata["format_version"] = "1.0.0"

        metadata[f"caption{caption_index+1}"] = caption

        for key, output in zip(keys, outputs):
            dtype = output.dtype  # long or one of float
            key_suffix = self.get_key_suffix(caption_index, dtype)
            tensor_dict[key + key_suffix] = output

            # remove lower precision latents if higher precision latents are already cached
            if overwrite:
                suffix_without_dtype = self.get_key_suffix(caption_index, None)
                remove_lower_precision_values(tensor_dict, [key + suffix_without_dtype])

        save_file(tensor_dict, cache_path, metadata=metadata)


class LatentsCachingStrategy:
    _strategy = None  # strategy instance: actual strategy class

    def __init__(
        self,
        architecture: str,
        latents_stride: int,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
    ) -> None:
        self._architecture = architecture
        self._latents_stride = latents_stride
        self._cache_to_disk = cache_to_disk
        self._batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check

        self.load_version_warning_printed = False
        self.save_version_warning_printed = False

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["LatentsCachingStrategy"]:
        return cls._strategy

    @property
    def architecture(self):
        return self._architecture

    @property
    def latents_stride(self):
        return self._latents_stride

    @property
    def cache_to_disk(self):
        return self._cache_to_disk

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def cache_suffix(self):
        return f"_{self.architecture.lower()}.safetensors"

    def get_image_size_from_disk_cache_path(
        self, absolute_path: str, cache_path: str
    ) -> Tuple[Optional[int], Optional[int]]:
        w, h = os.path.splitext(cache_path)[0].rsplit("_", 2)[-2].split("x")
        return int(w), int(h)

    def get_latents_cache_path_from_info(self, info: utils.ImageInfo) -> str:
        return self.get_latents_cache_path(info.absolute_path, info.image_size, info.latents_cache_dir)

    def get_latents_cache_path(
        self, absolute_path_or_archive_img_path: str, image_size: Tuple[int, int], cache_dir: Optional[str] = None
    ) -> str:
        if cache_dir is not None:
            if dataset_metadata_utils.is_archive_path(absolute_path_or_archive_img_path):
                inner_path = dataset_metadata_utils.get_inner_path(absolute_path_or_archive_img_path)
                archive_digest = dataset_metadata_utils.get_archive_digest(absolute_path_or_archive_img_path)
                cache_file_base = os.path.join(cache_dir, f"{archive_digest}_{inner_path}")
            else:
                cache_file_base = os.path.join(cache_dir, os.path.basename(absolute_path_or_archive_img_path))
        else:
            cache_file_base = absolute_path_or_archive_img_path

        return os.path.splitext(cache_file_base)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.cache_suffix

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        cache_path: str,
        flip_aug: bool,
        alpha_mask: bool,
        preferred_dtype: Optional[Union[str, torch.dtype]],
    ) -> bool:
        raise NotImplementedError

    def cache_batch_latents(self, model: Any, batch: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        raise NotImplementedError

    def get_key_suffix(
        self,
        bucket_reso: Optional[Tuple[int, int]] = None,
        latents_size: Optional[Tuple[int, int]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> str:
        """
        if dtype is None, it returns "_32x64" for example.
        """
        if latents_size is not None:
            expected_latents_size = latents_size  # H, W
        else:
            # bucket_reso is (W, H)
            expected_latents_size = (
                bucket_reso[1] // self.latents_stride,
                bucket_reso[0] // self.latents_stride,
            )  # H, W

        if dtype is None:
            dtype_suffix = ""
        else:
            dtype_suffix = "_" + utils.dtype_to_normalized_str(dtype)

        # e.g. "_32x64_float16", HxW, dtype
        key_suffix = f"_{expected_latents_size[0]}x{expected_latents_size[1]}{dtype_suffix}"

        return key_suffix

    def get_compatible_latents_keys(
        self,
        keys: set[str],
        dtype: Optional[Union[str, torch.dtype]],
        flip_aug: bool,
        bucket_reso: Optional[Tuple[int, int]] = None,
        latents_size: Optional[Tuple[int, int]] = None,
    ) -> list[Optional[str], Optional[str]]:
        """
        bucket_reso is (W, H), latents_size is (H, W)
        """

        key_suffix = self.get_key_suffix(bucket_reso, latents_size, None)
        keys_without_dtype = ["latents" + key_suffix]
        if flip_aug:
            keys_without_dtype.append("latents_flipped" + key_suffix)

        compatible_keys = get_compatible_dtype_keys(keys, keys_without_dtype, dtype)
        return (
            compatible_keys
            if flip_aug
            else compatible_keys[0] + [None] if isinstance(compatible_keys[0], list) else [compatible_keys[0], None]
        )

    def _default_is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        latents_cache_path: str,
        flip_aug: bool,
        alpha_mask: bool,
        preferred_dtype: Optional[Union[str, torch.dtype]],
    ):
        # multi_resolution is always enabled for any strategy
        if not self.cache_to_disk:
            return False
        if not os.path.exists(latents_cache_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        key_suffix_without_dtype = self.get_key_suffix(bucket_reso=bucket_reso, dtype=None)

        try:
            # safe_open locks the file, so we cannot use it for checking keys
            # with safe_open(latents_cache_path, framework="pt") as f:
            #     keys = f.keys()
            with utils.MemoryEfficientSafeOpen(latents_cache_path) as f:
                keys = f.keys()

            if alpha_mask and "alpha_mask" + key_suffix_without_dtype not in keys:
                # print(f"alpha_mask not found: {latents_cache_path}")
                return False

            # preferred_dtype is None if any dtype is acceptable
            latents_key, flipped_latents_key = self.get_compatible_latents_keys(
                keys, preferred_dtype, flip_aug, bucket_reso=bucket_reso
            )
            if latents_key is None or (flip_aug and flipped_latents_key is None):
                # print(f"Precise dtype not found: {latents_cache_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading file: {latents_cache_path}")
            raise e

        return True

    # TODO remove circular dependency for ImageInfo
    def _default_cache_batch_latents(
        self,
        encode_by_vae,
        vae_device,
        vae_dtype,
        image_infos: List[utils.ImageInfo],
        flip_aug: bool,
        alpha_mask: bool,
        random_crop: bool,
    ):
        """
        Default implementation for cache_batch_latents. Image loading, VAE, flipping, alpha mask handling are common.
        """
        from library import train_util  # import here to avoid circular import

        img_tensor, alpha_masks, original_sizes, crop_ltrbs = train_util.load_images_and_masks_for_caching(
            image_infos, alpha_mask, random_crop
        )
        img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latents_tensors = encode_by_vae(img_tensor).to("cpu")
        if flip_aug:
            img_tensor = torch.flip(img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = encode_by_vae(img_tensor).to("cpu")
        else:
            flipped_latents = [None] * len(latents_tensors)

        # for info, latents, flipped_latent, alpha_mask in zip(image_infos, latents_tensors, flipped_latents, alpha_masks):
        for i in range(len(image_infos)):
            info = image_infos[i]
            latents = latents_tensors[i]
            flipped_latent = flipped_latents[i]
            alpha_mask = alpha_masks[i]
            original_size = original_sizes[i]
            crop_ltrb = crop_ltrbs[i]

            if self.cache_to_disk:
                self.save_latents_to_disk(
                    info.latents_cache_path, latents, original_size, crop_ltrb, flipped_latent, alpha_mask
                )
            else:
                info.latents_original_size = original_size
                info.latents_crop_ltrb = crop_ltrb
                info.latents = latents
                if flip_aug:
                    info.latents_flipped = flipped_latent
                info.alpha_mask = alpha_mask

    def load_latents_from_disk(
        self, cache_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[torch.Tensor, List[int], List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        raise NotImplementedError

    def _default_load_latents_from_disk(
        self, cache_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[torch.Tensor, List[int], List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with safe_open(cache_path, framework="pt") as f:
            metadata = f.metadata()
            version = metadata.get("format_version", "0.0.0")
            major, minor, patch = map(int, version.split("."))
            if major > 1:  # or (major == 1 and minor > 0):
                if not self.load_version_warning_printed:
                    self.load_version_warning_printed = True
                    logger.warning(
                        f"Existing latents cache file has a higher version {version} for {cache_path}. This may cause issues."
                    )

            keys = f.keys()

            latents_key, flipped_latents_key = self.get_compatible_latents_keys(
                keys, None, flip_aug=True, bucket_reso=bucket_reso
            )

            key_suffix_without_dtype = self.get_key_suffix(bucket_reso=bucket_reso, dtype=None)
            alpha_mask_key = "alpha_mask" + key_suffix_without_dtype

            latents = f.get_tensor(latents_key)
            flipped_latents = f.get_tensor(flipped_latents_key) if flipped_latents_key is not None else None
            alpha_mask = f.get_tensor(alpha_mask_key) if alpha_mask_key in keys else None

            original_size = [int(metadata["width"]), int(metadata["height"])]
            crop_ltrb = metadata["crop_ltrb" + key_suffix_without_dtype]
            crop_ltrb = list(map(int, crop_ltrb.split(",")))

        return latents, original_size, crop_ltrb, flipped_latents, alpha_mask

    def save_latents_to_disk(
        self,
        cache_path: str,
        latents_tensor: torch.Tensor,
        original_size: Tuple[int, int],
        crop_ltrb: List[int],
        flipped_latents_tensor: Optional[torch.Tensor] = None,
        alpha_mask: Optional[torch.Tensor] = None,
    ):
        dtype = latents_tensor.dtype
        latents_size = latents_tensor.shape[1:3]  # H, W
        tensor_dict = {}

        overwrite = False
        if os.path.exists(cache_path):
            # load existing safetensors and update it
            overwrite = True

            # we cannot use safe_open here because it locks the file
            # with safe_open(cache_path, framework="pt") as f:
            with utils.MemoryEfficientSafeOpen(cache_path) as f:
                metadata = f.metadata()
                keys = f.keys()
                for key in keys:
                    tensor_dict[key] = f.get_tensor(key)
            assert metadata["architecture"] == self.architecture

            file_version = metadata.get("format_version", "0.0.0")
            major, minor, patch = map(int, file_version.split("."))
            if major > 1 or (major == 1 and minor > 0):
                self.save_version_warning_printed = True
                logger.warning(
                    f"Existing latents cache file has a higher version {file_version} for {cache_path}. This may cause issues."
                )
        else:
            metadata = {}
            metadata["architecture"] = self.architecture
            metadata["width"] = f"{original_size[0]}"
            metadata["height"] = f"{original_size[1]}"
            metadata["format_version"] = "1.0.0"

        metadata[f"crop_ltrb_{latents_size[0]}x{latents_size[1]}"] = ",".join(map(str, crop_ltrb))

        key_suffix = self.get_key_suffix(latents_size=latents_size, dtype=dtype)
        if latents_tensor is not None:
            tensor_dict["latents" + key_suffix] = latents_tensor
        if flipped_latents_tensor is not None:
            tensor_dict["latents_flipped" + key_suffix] = flipped_latents_tensor
        if alpha_mask is not None:
            key_suffix_without_dtype = self.get_key_suffix(latents_size=latents_size, dtype=None)
            tensor_dict["alpha_mask" + key_suffix_without_dtype] = alpha_mask

        # remove lower precision latents if higher precision latents are already cached
        if overwrite:
            suffix_without_dtype = self.get_key_suffix(latents_size=latents_size, dtype=None)
            remove_lower_precision_values(
                tensor_dict, ["latents" + suffix_without_dtype, "latents_flipped" + suffix_without_dtype]
            )

        save_file(tensor_dict, cache_path, metadata=metadata)
