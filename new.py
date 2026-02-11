#!/usr/bin/env python3
"""
Shiv Puran XML to Supabase Uploader
Parses local XML file and uploads chunks to Supabase
"""

import xml.etree.ElementTree as ET
import json
import re
from typing import List, Dict
import os

# Install required library first: pip install supabase
from supabase import create_client, Client

# ==================== CONFIGURATION ====================
# TODO: Replace these with your Supabase credentials
#Supabase keys
TABLE_NAME = "shiv_puran_chunks"  # Your table name

# Local file path
XML_FILE_PATH = r"D:\automated booksllm\dvoJ_sankshipt-shiv-puran-with-illustration-gita-press_daisy.xml"

# Chunk settings
CHUNK_SIZE = 1000  # Maximum characters per chunk
# =======================================================


class ShivPuranParser:
    def __init__(self, xml_file_path: str, chunk_size: int = 1000):
        self.xml_file_path = xml_file_path
        self.chunk_size = chunk_size
        self.namespace = {'dtb': 'http://www.daisy.org/z3986/2005/dtbook/'}
        
    def parse_xml(self) -> List[Dict]:
        """Parse the XML file and extract text content"""
        if not os.path.exists(self.xml_file_path):
            raise FileNotFoundError(f"XML file not found: {self.xml_file_path}")
        
        print(f"📖 Parsing XML file: {self.xml_file_path}")
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()
        
        chunks = []
        current_chunk = ""
        chunk_id = 1
        current_section = "Introduction"
        
        # Find all paragraphs
        for elem in root.iter():
            # Track section headers
            if elem.tag.endswith('h1') or elem.tag.endswith('h2') or elem.tag.endswith('h3'):
                if elem.text and elem.text.strip():
                    current_section = elem.text.strip()
            
            # Extract paragraph text
            if elem.tag.endswith('p'):
                text = elem.text if elem.text else ""
                text = text.strip()
                
                # Skip completely empty paragraphs
                if not text or len(text) < 1:
                    continue
                
                # Light cleaning - normalize whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                # If adding this text would exceed chunk size, save current chunk
                if len(current_chunk) + len(text) + 1 > self.chunk_size and current_chunk:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'section': current_section,
                        'content': current_chunk.strip(),
                        'char_count': len(current_chunk.strip())
                    })
                    chunk_id += 1
                    current_chunk = text
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n" + text
                    else:
                        current_chunk = text
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                'chunk_id': chunk_id,
                'section': current_section,
                'content': current_chunk.strip(),
                'char_count': len(current_chunk.strip())
            })
        
        print(f"✓ Created {len(chunks)} chunks")
        print(f"✓ Average chunk size: {sum(c['char_count'] for c in chunks) / len(chunks):.0f} characters")
        
        return chunks
    
    def upload_to_supabase(self, chunks: List[Dict], supabase_url: str, supabase_key: str, table_name: str):
        """Upload chunks to Supabase"""
        print(f"\n🚀 Connecting to Supabase...")
        
        try:
            # Create Supabase client
            supabase: Client = create_client(supabase_url, supabase_key)
            print("✓ Connected to Supabase")
            
            # Check if table exists by trying to query it
            try:
                test_query = supabase.table(table_name).select("*").limit(1).execute()
                print(f"✓ Table '{table_name}' found")
            except Exception as e:
                print(f"❌ Error accessing table '{table_name}': {e}")
                print("\nMake sure you've created the table with this SQL:")
                print(self.get_table_schema())
                return
            
            # Insert in batches to avoid timeouts
            batch_size = 100
            total_batches = (len(chunks) - 1) // batch_size + 1
            
            print(f"\n📤 Uploading {len(chunks)} chunks in {total_batches} batches...")
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    response = supabase.table(table_name).insert(batch).execute()
                    print(f"✓ Batch {batch_num}/{total_batches} uploaded ({len(batch)} chunks)")
                except Exception as e:
                    print(f"❌ Error uploading batch {batch_num}: {e}")
                    print(f"   First chunk in failed batch: {batch[0]}")
                    raise
            
            print(f"\n🎉 Successfully uploaded {len(chunks)} chunks to '{table_name}'!")
            
            # Verify upload
            result = supabase.table(table_name).select("count", count="exact").execute()
            print(f"✓ Verified: {result.count} total rows in table")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
    
    def get_table_schema(self):
        """Return the SQL schema for creating the table"""
        return """
CREATE TABLE shiv_puran_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL,
    section TEXT,
    content TEXT NOT NULL,
    char_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster searches
CREATE INDEX idx_chunk_id ON shiv_puran_chunks(chunk_id);
CREATE INDEX idx_section ON shiv_puran_chunks(section);
"""


def main():
    print("=" * 60)
    print("   Shiv Puran XML to Supabase Uploader")
    print("=" * 60)
    
    # Validate configuration
    if SUPABASE_URL == "https://your-project.supabase.co":
        print("\n❌ ERROR: Please update SUPABASE_URL in the script!")
        print("   Find it at: https://app.supabase.com → Your Project → Settings → API")
        return
    
    if SUPABASE_KEY == "your-anon-key-here":
        print("\n❌ ERROR: Please update SUPABASE_KEY in the script!")
        print("   Find it at: https://app.supabase.com → Your Project → Settings → API")
        return
    
    # Create parser
    parser = ShivPuranParser(XML_FILE_PATH, CHUNK_SIZE)
    
    try:
        # Parse XML
        chunks = parser.parse_xml()
        
        # Show preview
        print("\n📄 Preview of first 2 chunks:")
        for chunk in chunks[:2]:
            print(f"\n--- Chunk {chunk['chunk_id']} (Section: {chunk['section']}) ---")
            preview = chunk['content'][:150] + "..." if len(chunk['content']) > 150 else chunk['content']
            print(preview)
        
        # Upload to Supabase
        parser.upload_to_supabase(chunks, SUPABASE_URL, SUPABASE_KEY, TABLE_NAME)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print(f"\nPlease check the file path: {XML_FILE_PATH}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    main()
