#!/usr/bin/env python3
"""
Sanskrit to English Translation - Lightweight Models
Optimized for Sanskrit religious texts with minimal resource usage
"""

import time
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# ==================== LOAD ENV ====================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

# Create Supabase client once
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==================== CONFIGURATION ====================
TABLE_NAME = "shiv_puran_chunks"

TRANSLATION_MODEL = "m2m100-small"  # ai4bharat-small | nllb-small | m2m100-small
BATCH_SIZE = 10
START_FROM_CHUNK = 1
USE_GPU = False
MAX_LENGTH = 256
# =======================================================


class LightweightSanskritTranslator:
    """Lightweight translator optimized for Sanskrit"""

    def __init__(self, model_name: str = "ai4bharat-small", use_gpu: bool = False):
        print(f"\n📦 Loading {model_name} translation model...")

        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        print(f"   ✓ Using {self.device.upper()}")

        if model_name == "ai4bharat-small":
            model_id = "ai4bharat/indictrans2-en-indic-dist-200M"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, trust_remote_code=True
            )

            self.src_lang = "san_Deva"
            self.tgt_lang = "eng_Latn"

        elif model_name == "nllb-small":
            model_id = "facebook/nllb-200-distilled-600M"

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

            self.src_lang = "san_Deva"
            self.tgt_lang = "eng_Latn"

        elif model_name == "m2m100-small":
            model_id = "facebook/m2m100_418M"

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

            self.src_lang = "sa"
            self.tgt_lang = "en"

        else:
            raise ValueError(f"Unknown model: {model_name}")

        if self.device == "cuda":
            self.model = self.model.cuda()

        self.model.eval()
        self.model_name = model_name

        print("   ✓ Model loaded successfully")

    def translate(self, text: str) -> Optional[str]:
        try:
            import torch

            if self.model_name in ["nllb-small", "m2m100-small"]:
                self.tokenizer.src_lang = self.src_lang

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH
            )

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                if self.model_name == "nllb-small":
                    generated = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                        max_length=MAX_LENGTH,
                        num_beams=1
                    )
                elif self.model_name == "m2m100-small":
                    generated = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                        max_length=MAX_LENGTH,
                        num_beams=1
                    )
                else:
                    generated = self.model.generate(
                        **inputs,
                        max_length=MAX_LENGTH,
                        num_beams=1
                    )

            translation = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )[0]

            return translation.strip()

        except Exception as e:
            print(f"\n⚠️ Translation error: {e}")
            return None


class ShivPuranSanskritTranslator:

    def __init__(self, table_name: str, model_name: str, use_gpu: bool):
        self.supabase = supabase
        self.table_name = table_name
        self.translator = LightweightSanskritTranslator(model_name, use_gpu)

    def get_untranslated_chunks(self, start_from: int) -> List[Dict]:
        print(f"\n📖 Fetching untranslated chunks from {start_from}...")

        result = self.supabase.table(self.table_name) \
            .select("*") \
            .gte("chunk_id", start_from) \
            .is_("english_translation", None) \
            .order("chunk_id") \
            .execute()

        chunks = result.data
        print(f"✓ Found {len(chunks)} chunks")
        return chunks

    def translate_chunk(self, chunk: Dict) -> Optional[str]:
        content = chunk["content"].strip()

        if len(content) < 5:
            return None

        if len(content) > 800:
            sentences = content.replace("।", "।\n").replace("॥", "॥\n").split("\n")
            translations = []

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 5:
                    t = self.translator.translate(sentence)
                    if t:
                        translations.append(t)

            return " ".join(translations)

        return self.translator.translate(content)

    def update_translation(self, chunk_id: int, translation: str):
        self.supabase.table(self.table_name) \
            .update({"english_translation": translation}) \
            .eq("chunk_id", chunk_id) \
            .execute()

    def translate_all(self, start_from: int):
        print("\n" + "=" * 70)
        print("   Sanskrit → English Translation")
        print("=" * 70)

        chunks = self.get_untranslated_chunks(start_from)

        if not chunks:
            print("\n✅ Nothing to translate.")
            return

        total = len(chunks)
        translated = 0
        failed = 0
        start_time = time.time()

        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk["chunk_id"]

            try:
                print(f"[{i}/{total}] Chunk {chunk_id}...", end=" ", flush=True)

                translation = self.translate_chunk(chunk)

                if translation:
                    self.update_translation(chunk_id, translation)
                    translated += 1
                    print("✓")
                else:
                    print("⊘")

            except Exception as e:
                print(f"❌ {e}")
                failed += 1

        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("   Translation Complete")
        print("=" * 70)
        print(f"✅ Translated: {translated}")
        print(f"⚠️ Failed: {failed}")
        print(f"⏱ Time: {total_time/60:.1f} minutes")

        if translated > 0:
            print(f"📊 Avg: {total_time/translated:.2f}s per chunk")

        self.verify()

    def verify(self):
        print("\n🔍 Verifying...")

        translated = self.supabase.table(self.table_name) \
            .select("*", count="exact") \
            .not_.is_("english_translation", None) \
            .execute().count

        total = self.supabase.table(self.table_name) \
            .select("*", count="exact") \
            .execute().count

        percentage = (translated / total) * 100 if total else 0

        print(f"✓ {translated}/{total} translated ({percentage:.1f}%)")


def main():
    print("=" * 70)
    print("   Sanskrit Shiv Puran → English (Lightweight)")
    print("=" * 70)

    print("\n🔍 System Check:")
    try:
        import torch
        import transformers
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ Transformers {transformers.__version__}")
        print(f"✓ Using {'GPU' if USE_GPU and torch.cuda.is_available() else 'CPU'}")
    except ImportError:
        print("❌ Install required libraries:")
        print("pip install torch transformers sentencepiece")
        return

    try:
        translator = ShivPuranSanskritTranslator(
            TABLE_NAME,
            TRANSLATION_MODEL,
            USE_GPU
        )

        print("\n⚠️ Ensure column exists:")
        print("ALTER TABLE shiv_puran_chunks ADD COLUMN english_translation TEXT;")

        input("\nPress Enter to start...")

        translator.translate_all(START_FROM_CHUNK)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
