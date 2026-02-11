import os
from supabase import create_client, Client
from deep_translator import GoogleTranslator
import time
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------- LOAD SUPABASE CREDS FROM .env ----------
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

supabase: Client = create_client(URL, KEY)

# ---------- TRANSLATOR ----------
translator = GoogleTranslator(source="auto", target="en")

def translate_accurately(text: str, chunk_id: int) -> str:
    """
    Translate text with maximum accuracy
    Uses careful retry logic and quality checking
    """
    
    text = text.strip()
    
    # Handle very short content
    if len(text) < 3:
        return "[too short]"
    
    # If it's just symbols or numbers, mark it
    if text.replace(' ', '').replace('.', '').replace(',', '').isdigit():
        return f"[numeric: {text}]"
    
    best_translation = None
    max_attempts = 5
    
    for attempt in range(max_attempts):
        try:
            # Progressive delay to avoid rate limiting
            wait_time = 3 + (attempt * 2) + random.uniform(0, 2)
            time.sleep(wait_time)
            
            # For long text, split carefully
            if len(text) > 4000:
                # Split by Devanagari sentence enders
                parts = []
                current = ""
                
                for char in text:
                    current += char
                    if char in ['।', '॥', '\n'] and len(current) > 100:
                        parts.append(current.strip())
                        current = ""
                
                if current.strip():
                    parts.append(current.strip())
                
                # Translate each part
                translations = []
                for i, part in enumerate(parts):
                    if len(part) < 3:
                        continue
                    
                    try:
                        trans = translator.translate(part)
                        if trans and trans.strip() and trans != part:
                            translations.append(trans.strip())
                        
                        # Small delay between parts
                        if i < len(parts) - 1:
                            time.sleep(random.uniform(1, 2))
                    except Exception as e:
                        print(f"\n        Part {i+1} failed: {e}")
                        continue
                
                if translations:
                    result = ' '.join(translations)
                    if result and len(result) > 10:
                        best_translation = result
                        break
            
            else:
                # Direct translation for shorter text
                result = translator.translate(text)
                
                if result and result.strip():
                    # Quality check - make sure it's actually translated
                    if result != text and len(result) > 2:
                        best_translation = result.strip()
                        break
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "too many requests" in error_msg or "rate limit" in error_msg:
                wait = 20 + (attempt * 10)
                print(f"\n      ⚠️ Rate limit, waiting {wait}s...")
                time.sleep(wait)
            
            elif "connection" in error_msg:
                wait = 10 + (attempt * 5)
                print(f"\n      ⚠️ Connection issue, waiting {wait}s...")
                time.sleep(wait)
            
            else:
                if attempt < max_attempts - 1:
                    print(f"\n      ⚠️ Attempt {attempt+1} failed: {e}")
                    time.sleep(5)
                else:
                    print(f"\n      ❌ All attempts failed: {e}")
                    return None
    
    return best_translation

def translate_from_id(start_id=1801, end_id=None):
    """
    Translate all chunks from start_id onwards
    """
    
    print("="*70)
    print(f"   Accurate Translation - Starting from Chunk {start_id}")
    print("="*70)
    
    # Count chunks in range
    print(f"\n🔍 Checking chunks from ID {start_id} onwards...")
    
    query = supabase.table("shiv_puran_chunks") \
        .select("chunk_id", count="exact") \
        .gte("chunk_id", start_id)
    
    if end_id:
        query = query.lte("chunk_id", end_id)
    
    total_in_range = query.execute().count
    
    # Count how many are NULL
    null_query = supabase.table("shiv_puran_chunks") \
        .select("chunk_id", count="exact") \
        .gte("chunk_id", start_id) \
        .is_("English_translation", None)
    
    if end_id:
        null_query = null_query.lte("chunk_id", end_id)
    
    null_count = null_query.execute().count
    
    print(f"✓ Total chunks from {start_id}: {total_in_range}")
    print(f"✓ Chunks with NULL translation: {null_count}")
    
    if null_count == 0:
        print("\n✅ All chunks in this range already translated!")
        return
    
    # Show sample
    print("\n📋 Sample chunks to translate:")
    samples = supabase.table("shiv_puran_chunks") \
        .select("chunk_id, content") \
        .gte("chunk_id", start_id) \
        .is_("English_translation", None) \
        .limit(3) \
        .execute()
    
    for s in samples.data:
        preview = s['content'][:80] if s['content'] else '[empty]'
        print(f"   Chunk {s['chunk_id']}: {preview}...")
    
    print(f"\n📝 Will translate {null_count} chunks")
    print(f"⏱️  Estimated time: {(null_count * 5) / 60:.1f} minutes")
    print("   ⚠️  Using high-quality, careful mode")
    
    choice = input(f"\nTranslate all NULL chunks from ID {start_id}? (y/n): ")
    if choice.lower() != 'y':
        print("Cancelled.")
        return
    
    # Process chunks
    translated = 0
    skipped = 0
    failed = 0
    start_time = time.time()
    
    # Get all NULL chunks in range
    all_null_chunks = supabase.table("shiv_puran_chunks") \
        .select("chunk_id, content") \
        .gte("chunk_id", start_id) \
        .is_("English_translation", None) \
        .order("chunk_id")
    
    if end_id:
        all_null_chunks = all_null_chunks.lte("chunk_id", end_id)
    
    chunks = all_null_chunks.execute().data
    
    total = len(chunks)
    print(f"\n🚀 Starting translation of {total} chunks...\n")
    
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        
        # Progress indicator
        percentage = (i / total) * 100
        print(f"[{i}/{total} - {percentage:.1f}%] Chunk {chunk_id}...", end=" ", flush=True)
        
        # Handle empty/None content
        if not content or not content.strip():
            print("⊘ (empty)")
            try:
                supabase.table("shiv_puran_chunks") \
                    .update({"English_translation": "[empty]"}) \
                    .eq("chunk_id", chunk_id) \
                    .execute()
                skipped += 1
            except:
                pass
            continue
        
        content_clean = content.strip()
        
        # Handle very short
        if len(content_clean) < 3:
            print(f"⊘ (too short: {len(content_clean)} chars)")
            try:
                supabase.table("shiv_puran_chunks") \
                    .update({"English_translation": "[too short]"}) \
                    .eq("chunk_id", chunk_id) \
                    .execute()
                skipped += 1
            except:
                pass
            continue
        
        # Translate
        try:
            chunk_start = time.time()
            translation = translate_accurately(content_clean, chunk_id)
            chunk_time = time.time() - chunk_start
            
            if translation and translation.strip():
                # Update database
                supabase.table("shiv_puran_chunks") \
                    .update({"English_translation": translation}) \
                    .eq("chunk_id", chunk_id) \
                    .execute()
                
                translated += 1
                print(f"✓ ({chunk_time:.1f}s)")
                
                # Show preview
                preview = translation[:70] + "..." if len(translation) > 70 else translation
                print(f"    → {preview}")
            else:
                print("❌ (failed)")
                failed += 1
                time.sleep(10)  # Wait longer after failure
        
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
            time.sleep(10)
        
        # Progress update every 20 chunks
        if i % 20 == 0:
            elapsed = time.time() - start_time
            avg_per_chunk = elapsed / i
            remaining = (total - i) * avg_per_chunk
            
            print(f"\n  📊 Progress Update:")
            print(f"     Completed: {i}/{total} ({percentage:.1f}%)")
            print(f"     Translated: {translated}, Skipped: {skipped}, Failed: {failed}")
            print(f"     Time: {elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining")
            print(f"     Avg speed: {avg_per_chunk:.1f}s per chunk\n")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("   TRANSLATION COMPLETE!")
    print("="*70)
    print(f"✅ Successfully translated: {translated}")
    print(f"⊘ Skipped (empty/short): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    if translated > 0:
        print(f"📊 Average: {total_time/translated:.2f}s per chunk")
    
    # Verify
    verify_range(start_id, end_id)

def verify_range(start_id=1801, end_id=None):
    """Verify translation status for a range"""
    
    print("\n" + "="*70)
    print(f"   VERIFICATION (Chunks from {start_id})")
    print("="*70)
    
    # Total in range
    total_query = supabase.table("shiv_puran_chunks") \
        .select("chunk_id", count="exact") \
        .gte("chunk_id", start_id)
    
    if end_id:
        total_query = total_query.lte("chunk_id", end_id)
    
    total = total_query.execute().count
    
    # Has translation
    has_trans_query = supabase.table("shiv_puran_chunks") \
        .select("chunk_id", count="exact") \
        .gte("chunk_id", start_id) \
        .not_.is_("English_translation", None)
    
    if end_id:
        has_trans_query = has_trans_query.lte("chunk_id", end_id)
    
    has_trans = has_trans_query.execute().count
    
    # Still NULL
    still_null_query = supabase.table("shiv_puran_chunks") \
        .select("chunk_id", count="exact") \
        .gte("chunk_id", start_id) \
        .is_("English_translation", None)
    
    if end_id:
        still_null_query = still_null_query.lte("chunk_id", end_id)
    
    still_null = still_null_query.execute().count
    
    percentage = (has_trans / total) * 100 if total > 0 else 0
    
    print(f"\n📊 Status for chunks {start_id}+:")
    print(f"   Total: {total}")
    print(f"   ✅ Translated: {has_trans} ({percentage:.1f}%)")
    print(f"   ❌ Still NULL: {still_null}")
    
    if still_null > 0:
        print(f"\n⚠️  {still_null} chunks still need translation")
        print("   Run this script again to retry them")
    else:
        print("\n🎉 SUCCESS! All chunks from {start_id} are now translated!")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   Shiv Puran - Accurate Translation Tool")
    print("="*70)
    
    print("\nThis will translate all NULL chunks starting from a specific ID")
    
    try:
        start = input("\nEnter starting chunk_id (default: 1801): ").strip()
        start_id = int(start) if start else 1801
        
        end = input("Enter ending chunk_id (press Enter for all remaining): ").strip()
        end_id = int(end) if end else None
        
        translate_from_id(start_id, end_id)
        
    except ValueError:
        print("Invalid ID. Please enter a number.")
    except KeyboardInterrupt:

        print("\n\nStopped by user.")

