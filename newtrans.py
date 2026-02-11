#!/usr/bin/env python3
"""
Sanskrit to English Translation - Lightweight Models
Optimized for Sanskrit religious texts with minimal resource usage
"""

import time
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv  # NEW

# Load environment variables from .env file
load_dotenv()
from supabase import create_client, Client

# ==================== CONFIGURATION ====================
# Supabase Configuration
#keys
TABLE_NAME = "shiv_puran_chunks"  # Your table name
# LIGHTWEIGHT MODEL OPTIONS (Choose ONE):
# 
# 1. "ai4bharat-small" - AI4Bharat IndicTrans2 (BEST for Sanskrit) ⭐⭐⭐
#    Size: ~300MB | Quality: 8.5/10 | Speed: Very Fast | RAM: 2GB
#    Specifically trained on Sanskrit, Devanagari script
#
# 2. "nllb-small" - NLLB Distilled (Good general model) ⭐⭐
#    Size: 600MB | Quality: 7/10 | Speed: Fast | RAM: 2-3GB
#
# 3. "m2m100-small" - M2M100 418M (Lightweight multilingual) ⭐
#    Size: 418MB | Quality: 6.5/10 | Speed: Very Fast | RAM: 2GB
#
TRANSLATION_MODEL = "m2m100-small"  # No authentication needed!  # RECOMMENDED for Sanskrit!

# Translation Settings
BATCH_SIZE = 10  # Process multiple chunks (lightweight models can handle more)
START_FROM_CHUNK = 1
USE_GPU = False  # These models work great on CPU!
MAX_LENGTH = 256  # Shorter = faster
# =======================================================


class LightweightSanskritTranslator:
    """Lightweight translator optimized for Sanskrit"""
    
    def __init__(self, model_name: str = "ai4bharat-small", use_gpu: bool = False):
        print(f"\n📦 Loading {model_name} translation model...")
        print("   Optimized for Sanskrit → English")
        
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Use CPU by default for these lightweight models
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            print("   ✓ Using GPU")
        else:
            print("   ✓ Using CPU (recommended for lightweight models)")
        
        if model_name == "ai4bharat-small":
            # AI4Bharat IndicTrans2 - BEST for Sanskrit!
            # Specifically designed for Indian languages including Sanskrit
            print("   Downloading AI4Bharat IndicTrans2-En (Distilled)...")
            
            model_id = "ai4bharat/indictrans2-en-indic-dist-200M"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self.src_lang = "san_Deva"  # Sanskrit in Devanagari
            self.tgt_lang = "eng_Latn"  # English
            
            print(f"   ✓ Loaded AI4Bharat IndicTrans2 (~200MB)")
            print(f"   ✓ Optimized for Sanskrit Devanagari script")
            
        elif model_name == "nllb-small":
            # NLLB Distilled 600M
            print("   Downloading NLLB-200 Distilled...")
            
            model_id = "facebook/nllb-200-distilled-600M"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self.src_lang = "san_Deva"  # Sanskrit
            self.tgt_lang = "eng_Latn"  # English
            
            print(f"   ✓ Loaded NLLB-200 (~600MB)")
            
        elif model_name == "m2m100-small":
            # M2M100 418M - Very lightweight
            print("   Downloading M2M100 418M...")
            
            model_id = "facebook/m2m100_418M"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self.src_lang = "sa"  # Sanskrit
            self.tgt_lang = "en"  # English
            
            print(f"   ✓ Loaded M2M100 (~418MB)")
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
        
        # Put model in eval mode for faster inference
        self.model.eval()
    
    def translate(self, text: str) -> str:
        """Translate Sanskrit text to English"""
        try:
            import torch
            
            if self.model_name == "ai4bharat-small":
                # AI4Bharat IndicTrans2 method
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate with optimized settings for speed
                with torch.no_grad():  # Faster inference
                    generated = self.model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=1,  # Greedy decoding = fastest
                        early_stopping=True
                    )
                
                translation = self.tokenizer.batch_decode(
                    generated, 
                    skip_special_tokens=True
                )[0]
                
            elif self.model_name == "nllb-small":
                # NLLB method
                self.tokenizer.src_lang = self.src_lang
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                        max_length=256,
                        num_beams=1
                    )
                
                translation = self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True
                )[0]
                
            elif self.model_name == "m2m100-small":
                # M2M100 method
                self.tokenizer.src_lang = self.src_lang
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                        max_length=256,
                        num_beams=1
                    )
                
                translation = self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True
                )[0]
            
            return translation.strip()
            
        except Exception as e:
            print(f"\n      ⚠️ Translation error: {e}")
            return None


class ShivPuranSanskritTranslator:
    """Main translator class"""
    
    def __init__(self, supabase_url: str, supabase_key: str, table_name: str,
                 model_name: str, use_gpu: bool):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.table_name = table_name
        self.translator = LightweightSanskritTranslator(model_name, use_gpu)
    
    def get_untranslated_chunks(self, start_from: int = 1) -> List[Dict]:
        """Get chunks that need translation"""
        print(f"\n📖 Fetching untranslated chunks from chunk_id {start_from}...")
        
        result = self.supabase.table(self.table_name)\
            .select("*")\
            .gte("chunk_id", start_from)\
            .is_("english_translation", "null")\
            .order("chunk_id")\
            .execute()
        
        chunks = result.data
        print(f"✓ Found {len(chunks)} chunks to translate")
        return chunks
    
    def translate_chunk(self, chunk: Dict) -> Optional[str]:
        """Translate a single chunk"""
        content = chunk['content'].strip()
        
        # Skip very short content
        if len(content) < 5:
            return None
        
        # For long content, split into sentences
        if len(content) > 800:
            # Split by common Devanagari sentence enders
            sentences = content.replace('।', '।\n').replace('॥', '॥\n').split('\n')
            translations = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 5:
                    trans = self.translator.translate(sentence)
                    if trans:
                        translations.append(trans)
            
            return ' '.join(translations)
        else:
            return self.translator.translate(content)
    
    def update_translation(self, chunk_id: int, translation: str):
        """Update translation in database"""
        self.supabase.table(self.table_name)\
            .update({"english_translation": translation})\
            .eq("chunk_id", chunk_id)\
            .execute()
    
    def translate_all(self, batch_size: int, start_from: int):
        """Translate all chunks"""
        print("\n" + "="*70)
        print("   Sanskrit to English Translation - Lightweight & Fast")
        print("="*70)
        
        chunks = self.get_untranslated_chunks(start_from)
        
        if not chunks:
            print("\n✅ All chunks already translated!")
            return
        
        total = len(chunks)
        print(f"\n📝 Translating {total} chunks")
        print(f"   Model: {TRANSLATION_MODEL}")
        print(f"   Batch size: {batch_size}")
        
        # Speed estimate (lightweight models are fast!)
        est_per_chunk = 1.5 if USE_GPU else 3  # seconds
        est_minutes = (total * est_per_chunk) / 60
        print(f"   Estimated time: {est_minutes:.1f} minutes")
        
        translated = 0
        failed = 0
        start_time = time.time()
        
        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk['chunk_id']
            
            try:
                print(f"[{i}/{total}] Chunk {chunk_id}...", end=" ", flush=True)
                
                chunk_start = time.time()
                translation = self.translate_chunk(chunk)
                chunk_time = time.time() - chunk_start
                
                if translation:
                    self.update_translation(chunk_id, translation)
                    translated += 1
                    print(f"✓ ({chunk_time:.1f}s)")
                    
                    # Show preview
                    preview = translation[:70] + "..." if len(translation) > 70 else translation
                    print(f"    → {preview}")
                else:
                    print("⊘")
                    
            except Exception as e:
                print(f"❌ {e}")
                failed += 1
            
            # Progress update
            if i % 25 == 0:
                elapsed = time.time() - start_time
                avg = elapsed / i
                remaining = (total - i) * avg
                print(f"\n  📊 Progress: {(i/total)*100:.1f}% | {translated} done | ~{remaining/60:.1f}m left\n")
        
        # Summary
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("   Translation Complete!")
        print("="*70)
        print(f"✅ Translated: {translated}")
        print(f"⚠️  Failed: {failed}")
        print(f"⏱️  Time: {total_time/60:.1f} minutes")
        print(f"📊 Average: {total_time/translated:.2f}s per chunk")
        
        self.verify()
    
    def verify(self):
        """Verify translation count"""
        print("\n🔍 Verifying...")
        
        result = self.supabase.table(self.table_name)\
            .select("*", count="exact")\
            .not_.is_("english_translation", "null")\
            .execute()
        
        translated = result.count
        
        total_result = self.supabase.table(self.table_name)\
            .select("*", count="exact")\
            .execute()
        
        total = total_result.count
        percentage = (translated / total) * 100
        
        print(f"✓ {translated}/{total} chunks translated ({percentage:.1f}%)")


def main():
    print("="*70)
    print("   Sanskrit Shiv Puran → English (Lightweight Models)")
    print("="*70)
    
    # Validate config
    if SUPABASE_URL == "https://your-project.supabase.co":
        print("\n❌ Update SUPABASE_URL in script!")
        return
    
    if SUPABASE_KEY == "your-anon-key-here":
        print("\n❌ Update SUPABASE_KEY in script!")
        return
    
    # System check
    print("\n🔍 System Check:")
    try:
        import torch
        import transformers
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ Transformers {transformers.__version__}")
        
        if USE_GPU and torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"✓ Using CPU (perfect for lightweight models!)")
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("Run: pip install torch transformers sentencepiece")
        return
    
    try:
        translator = ShivPuranSanskritTranslator(
            SUPABASE_URL,
            SUPABASE_KEY,
            TABLE_NAME,
            TRANSLATION_MODEL,
            USE_GPU
        )
        
        print("\n⚠️  Make sure you've added the column:")
        print("   ALTER TABLE shiv_puran_chunks ADD COLUMN english_translation TEXT;")
        input("\nPress Enter to start translation...")
        
        translator.translate_all(BATCH_SIZE, START_FROM_CHUNK)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    main()

