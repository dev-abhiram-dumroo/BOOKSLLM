#!/usr/bin/env python3
"""
Shiv Puran XML to Supabase Uploader
Parses local XML file and uploads chunks to Supabase
"""

import xml.etree.ElementTree as ET
import re
from typing import List, Dict
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# ==================== LOAD ENV VARIABLES ====================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

# Create Supabase client once
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==================== CONFIGURATION ====================
TABLE_NAME = "shiv_puran_chunks"

XML_FILE_PATH = r"your path"

CHUNK_SIZE = 1000
# =======================================================


class ShivPuranParser:
    def __init__(self, xml_file_path: str, chunk_size: int = 1000):
        self.xml_file_path = xml_file_path
        self.chunk_size = chunk_size

    def parse_xml(self) -> List[Dict]:
        """Parse XML file and create text chunks"""

        if not os.path.exists(self.xml_file_path):
            raise FileNotFoundError(f"XML file not found: {self.xml_file_path}")

        print(f"📖 Parsing XML file: {self.xml_file_path}")

        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()

        chunks = []
        current_chunk = ""
        chunk_id = 1
        current_section = "Introduction"

        for elem in root.iter():

            # Track section headers
            if elem.tag.endswith(("h1", "h2", "h3")):
                if elem.text and elem.text.strip():
                    current_section = elem.text.strip()

            # Extract paragraph text
            if elem.tag.endswith("p"):
                text = elem.text.strip() if elem.text else ""

                if not text:
                    continue

                text = re.sub(r"\s+", " ", text)

                # If adding this paragraph exceeds chunk size, save current chunk
                if len(current_chunk) + len(text) + 1 > self.chunk_size and current_chunk:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "section": current_section,
                        "content": current_chunk.strip(),
                        "char_count": len(current_chunk.strip())
                    })
                    chunk_id += 1
                    current_chunk = text
                else:
                    current_chunk += ("\n" if current_chunk else "") + text

        # Save final chunk
        if current_chunk.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "section": current_section,
                "content": current_chunk.strip(),
                "char_count": len(current_chunk.strip())
            })

        print(f"✓ Created {len(chunks)} chunks")

        if chunks:
            avg = sum(c["char_count"] for c in chunks) / len(chunks)
            print(f"✓ Average chunk size: {avg:.0f} characters")

        return chunks

    def upload_to_supabase(self, chunks: List[Dict], table_name: str):
        """Upload chunks to Supabase"""

        print("\n🚀 Uploading to Supabase...")

        # Test table access
        try:
            supabase.table(table_name).select("*").limit(1).execute()
            print(f"✓ Table '{table_name}' found")
        except Exception as e:
            print(f"❌ Cannot access table '{table_name}': {e}")
            print("\nCreate table using this SQL:\n")
            print(self.get_table_schema())
            return

        batch_size = 100
        total_batches = (len(chunks) - 1) // batch_size + 1

        print(f"\n📤 Uploading {len(chunks)} chunks in {total_batches} batches...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1

            try:
                supabase.table(table_name).insert(batch).execute()
                print(f"✓ Batch {batch_num}/{total_batches} uploaded ({len(batch)} chunks)")
            except Exception as e:
                print(f"❌ Error uploading batch {batch_num}: {e}")
                raise

        print(f"\n🎉 Successfully uploaded {len(chunks)} chunks!")

        # Verify
        result = supabase.table(table_name).select("chunk_id", count="exact").execute()
        print(f"✓ Verified: {result.count} total rows in table")

    def get_table_schema(self):
        return """
CREATE TABLE shiv_puran_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL,
    section TEXT,
    content TEXT NOT NULL,
    char_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_chunk_id ON shiv_puran_chunks(chunk_id);
CREATE INDEX idx_section ON shiv_puran_chunks(section);
"""


def main():
    print("=" * 60)
    print("   Shiv Puran XML to Supabase Uploader")
    print("=" * 60)

    parser = ShivPuranParser(XML_FILE_PATH, CHUNK_SIZE)

    try:
        # Parse XML
        chunks = parser.parse_xml()

        # Preview
        print("\n📄 Preview of first 2 chunks:")
        for chunk in chunks[:2]:
            print(f"\n--- Chunk {chunk['chunk_id']} (Section: {chunk['section']}) ---")
            preview = chunk["content"][:150]
            print(preview + ("..." if len(chunk["content"]) > 150 else ""))

        confirm = input("\nUpload to Supabase? (y/n): ").strip().lower()
        if confirm != "y":
            print("Upload cancelled.")
            return

        # Upload
        parser.upload_to_supabase(chunks, TABLE_NAME)

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
