import os
import torch
from transformers import BertJapaneseTokenizer

from models.bert import BertForSeq2SeqDecoder, BertConfig
from utils.loader import batch_list_to_batch_tensors, Preprocess4Seq2seqDecoder
from utils.text import detokenize, TextCleaner

class TitleGenerator:
    def __init__(self, path, beam_size=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.beam_size = beam_size
        length_penalty = 0

        max_seq_length=512
        max_tgt_length=48
        ngram_size=3
        min_len=1
        forbid_duplicate_ngrams=True
        mode='s2s'
        do_lower_case=True
        pos_shift=False

        config_file = os.path.join(path, "config.json")

        self.tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", do_lower_case=do_lower_case)
        self.tokenizer.max_len = max_seq_length
        self.text_cleaner = TextCleaner

        print(f"load config : {config_file}")
        config = BertConfig.from_json_file(config_file)

        self.proc = Preprocess4Seq2seqDecoder(
            list(self.tokenizer.vocab.keys()), self.tokenizer.convert_tokens_to_ids, max_seq_length,
            max_tgt_length=max_tgt_length, pos_shift=pos_shift,
            source_type_id=config.source_type_id, target_type_id=config.target_type_id,
            cls_token=self.tokenizer.cls_token, sep_token=self.tokenizer.sep_token, pad_token=self.tokenizer.pad_token)

        mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token, self.tokenizer.sep_token, self.tokenizer.sep_token])

        forbid_ignore_set = None

        self.model = BertForSeq2SeqDecoder.from_pretrained(
            path, config=config, mask_word_id=mask_word_id, search_beam_size=self.beam_size,
            length_penalty=length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
            ngram_size=ngram_size, min_len=min_len, mode=mode,
            max_position_embeddings=max_seq_length, pos_shift=pos_shift,
        )
        self.model.to(self.device)
        self.model.eval()

        self.max_src_length = max_seq_length - 2 - max_tgt_length

    def generate(self, text):
        title = None

        try:
            source_text = self.text_cleaner.clean(text)
        except:
            return False, title

        if len(source_text) == 0:
            return False, title

        try:
            source_tokens = self.tokenizer.tokenize(source_text)[:self.max_src_length]
            tokenized_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)

            max_a_len = len(source_tokens)
            instances = self.proc((source_tokens, max_a_len))

            with torch.no_grad():
                # --- preprocess ---
                # single batch
                batch = batch_list_to_batch_tensors([instances]) # 6, batch_size, max_a_len
                batch = [t.to(self.device) if t is not None else None for t in batch]
                input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch

                # --- inference ---
                # dict : 'pred_seq', 'scores', 'wids', 'ptrs'
                traces = self.model(input_ids, token_type_ids, position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)

                # --- postprocess ---
                if self.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                else:
                    output_ids = traces.tolist()
                w_ids = output_ids[0] # 推論結果のID
                output_buf = self.tokenizer.convert_ids_to_tokens(w_ids) # tokenに変換
                # [SEP][PAD]の除去
                output_tokens = []
                for t in output_buf:
                    if t in (self.tokenizer.sep_token, self.tokenizer.pad_token):
                        break
                    output_tokens.append(t)

                # tokenの結合
                title = ''.join(detokenize(output_tokens))

            return True, title
        except:
            return False, title