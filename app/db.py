import os
import json

# Load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

USE_POSTGRES = bool(os.environ.get('DATABASE_URL'))

if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor
else:
    import sqlite3

def get_connection():
    """Get database connection (PostgreSQL on Replit, SQLite locally)."""
    if USE_POSTGRES:
        database_url = os.environ.get('DATABASE_URL')
        
        # Check if we have individual connection params (for local dev)
        pg_host = os.environ.get('PGHOST', 'localhost')
        pg_port = os.environ.get('PGPORT', '5432')
        pg_user = os.environ.get('PGUSER', 'postgres')
        pg_password = os.environ.get('PGPASSWORD', '')
        pg_database = os.environ.get('PGDATABASE', 'pdf_extraction')
        
        # If individual params are set, use them (more reliable on Windows)
        if os.environ.get('PGPASSWORD'):
            return psycopg2.connect(
                host=pg_host,
                port=pg_port,
                user=pg_user,
                password=pg_password,
                database=pg_database
            )
        else:
            return psycopg2.connect(database_url)
    else:
        db_path = os.path.join(os.path.dirname(__file__), 'data', 'pdf_extraction.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return sqlite3.connect(db_path)

def init_db(db_path=None):
    """Initialize database with required tables.
    
    Args:
        db_path: Optional path for SQLite (ignored for PostgreSQL)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    if USE_POSTGRES:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id SERIAL PRIMARY KEY,
                run_name TEXT UNIQUE NOT NULL,
                file_name TEXT,
                page_count INTEGER,
                figure_count INTEGER,
                table_count INTEGER,
                section_count INTEGER,
                ocr_pages INTEGER,
                timings_json TEXT,
                created_at TEXT
            )
        ''')
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id SERIAL PRIMARY KEY,
                run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
                page_number INTEGER,
                raw_text TEXT,
                word_count INTEGER,
                char_count INTEGER,
                ocr_used BOOLEAN DEFAULT FALSE,
                quality_metrics_json TEXT
            )
        ''')
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS figures (
                id SERIAL PRIMARY KEY,
                run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
                figure_id TEXT,
                page_number INTEGER,
                bbox_json TEXT,
                image_path TEXT,
                caption TEXT
            )
        ''')
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS tables (
                id SERIAL PRIMARY KEY,
                run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
                table_id TEXT,
                page_number INTEGER,
                bbox_json TEXT,
                rows INTEGER,
                cols INTEGER,
                cells_json TEXT,
                image_path TEXT
            )
        ''')
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS sections (
                id SERIAL PRIMARY KEY,
                run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
                section_id TEXT,
                title TEXT,
                pages_json TEXT,
                paragraphs_json TEXT,
                figure_count INTEGER,
                table_count INTEGER,
                raw_text TEXT
            )
        ''')
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id SERIAL PRIMARY KEY,
                section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
                summary_type TEXT,
                content TEXT,
                bullets_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    else:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT UNIQUE NOT NULL,
                file_name TEXT,
                page_count INTEGER,
                figure_count INTEGER,
                table_count INTEGER,
                section_count INTEGER,
                ocr_pages INTEGER,
                timings_json TEXT,
                created_at TEXT
            )
        ''')
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
                page_number INTEGER,
                raw_text TEXT,
                word_count INTEGER,
                char_count INTEGER,
                ocr_used INTEGER DEFAULT 0,
                quality_metrics_json TEXT
            )
        ''')
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS figures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
                figure_id TEXT,
                page_number INTEGER,
                bbox_json TEXT,
                image_path TEXT,
                caption TEXT
            )
        ''')
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
                table_id TEXT,
                page_number INTEGER,
                bbox_json TEXT,
                rows INTEGER,
                cols INTEGER,
                cells_json TEXT,
                image_path TEXT
            )
        ''')
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
                section_id TEXT,
                title TEXT,
                pages_json TEXT,
                paragraphs_json TEXT,
                figure_count INTEGER,
                table_count INTEGER,
                raw_text TEXT
            )
        ''')
        
        cur.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
                summary_type TEXT,
                content TEXT,
                bullets_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    conn.commit()
    cur.close()
    conn.close()

def save_run(db_path, run_name, meta, pages, figures, tables, sections):
    """Save extraction run data to database.
    
    Args:
        db_path: Ignored (kept for backward compatibility)
        run_name: Unique identifier for this run
        meta: Dictionary with run metadata
        pages: List of page dictionaries
        figures: List of figure dictionaries
        tables: List of table dictionaries
        sections: List of section dictionaries
    """
    conn = get_connection()
    cur = conn.cursor()
    
    param = '%s' if USE_POSTGRES else '?'
    
    try:
        if USE_POSTGRES:
            cur.execute(f'''
                INSERT INTO runs (run_name, file_name, page_count, figure_count, 
                                table_count, section_count, ocr_pages, timings_json, created_at)
                VALUES ({param}, {param}, {param}, {param}, {param}, {param}, {param}, {param}, {param})
                ON CONFLICT (run_name) DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    page_count = EXCLUDED.page_count,
                    figure_count = EXCLUDED.figure_count,
                    table_count = EXCLUDED.table_count,
                    section_count = EXCLUDED.section_count,
                    ocr_pages = EXCLUDED.ocr_pages,
                    timings_json = EXCLUDED.timings_json,
                    created_at = EXCLUDED.created_at
                RETURNING id
            ''', (
                run_name,
                meta.get('file', ''),
                meta.get('pages', 0),
                meta.get('figures', 0),
                meta.get('tables', 0),
                meta.get('sections', 0),
                meta.get('ocr_pages', 0),
                json.dumps(meta.get('timings_s', {})),
                meta.get('created_at', '')
            ))
            run_id = cur.fetchone()[0]
        else:
            cur.execute(f'''
                INSERT OR REPLACE INTO runs (run_name, file_name, page_count, figure_count, 
                                table_count, section_count, ocr_pages, timings_json, created_at)
                VALUES ({param}, {param}, {param}, {param}, {param}, {param}, {param}, {param}, {param})
            ''', (
                run_name,
                meta.get('file', ''),
                meta.get('pages', 0),
                meta.get('figures', 0),
                meta.get('tables', 0),
                meta.get('sections', 0),
                meta.get('ocr_pages', 0),
                json.dumps(meta.get('timings_s', {})),
                meta.get('created_at', '')
            ))
            cur.execute(f'SELECT id FROM runs WHERE run_name = {param}', (run_name,))
            run_id = cur.fetchone()[0]
        
        cur.execute(f'DELETE FROM pages WHERE run_id = {param}', (run_id,))
        cur.execute(f'DELETE FROM figures WHERE run_id = {param}', (run_id,))
        cur.execute(f'DELETE FROM tables WHERE run_id = {param}', (run_id,))
        cur.execute(f'DELETE FROM sections WHERE run_id = {param}', (run_id,))
        
        for page in pages:
            ocr_value = page.get('ocr_used', False)
            if USE_POSTGRES:
                ocr_value = bool(ocr_value)
            else:
                ocr_value = 1 if ocr_value else 0
            cur.execute(f'''
                INSERT INTO pages (run_id, page_number, raw_text, word_count, 
                                  char_count, ocr_used, quality_metrics_json)
                VALUES ({param}, {param}, {param}, {param}, {param}, {param}, {param})
            ''', (
                run_id,
                page.get('page', 0),
                page.get('text', ''),
                page.get('word_count', 0),
                page.get('char_count', 0),
                ocr_value,
                json.dumps(page.get('quality_metrics', {}))
            ))
        
        for fig in figures:
            cur.execute(f'''
                INSERT INTO figures (run_id, figure_id, page_number, bbox_json, 
                                    image_path, caption)
                VALUES ({param}, {param}, {param}, {param}, {param}, {param})
            ''', (
                run_id,
                fig.get('id', ''),
                fig.get('page', 0),
                json.dumps(fig.get('bbox', [])),
                fig.get('image', ''),
                fig.get('caption', '')
            ))
        
        for tbl in tables:
            cur.execute(f'''
                INSERT INTO tables (run_id, table_id, page_number, bbox_json, 
                                   rows, cols, cells_json, image_path)
                VALUES ({param}, {param}, {param}, {param}, {param}, {param}, {param}, {param})
            ''', (
                run_id,
                tbl.get('id', ''),
                tbl.get('page', 0),
                json.dumps(tbl.get('bbox', [])),
                tbl.get('rows', 0),
                tbl.get('cols', 0),
                json.dumps(tbl.get('cells', [])),
                tbl.get('image', '')
            ))
        
        for sec in sections:
            cur.execute(f'''
                INSERT INTO sections (run_id, section_id, title, pages_json, 
                                     paragraphs_json, figure_count, table_count, raw_text)
                VALUES ({param}, {param}, {param}, {param}, {param}, {param}, {param}, {param})
            ''', (
                run_id,
                sec.get('id', ''),
                sec.get('title', ''),
                json.dumps(sec.get('pages', [])),
                json.dumps(sec.get('paragraphs', [])),
                sec.get('figure_count', 0),
                sec.get('table_count', 0),
                sec.get('raw_text', '')
            ))
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def get_all_runs():
    """Get all extraction runs from the database."""
    conn = get_connection()
    cur = conn.cursor()
    
    if USE_POSTGRES:
        cur = conn.cursor(cursor_factory=RealDictCursor)
    else:
        cur.row_factory = sqlite3.Row
        cur = conn.cursor()
    
    cur.execute('SELECT * FROM runs ORDER BY created_at DESC')
    runs = cur.fetchall()
    
    cur.close()
    conn.close()
    return [dict(r) for r in runs]

def get_run_data(run_name):
    """Get complete data for a specific run."""
    conn = get_connection()
    param = '%s' if USE_POSTGRES else '?'
    
    if USE_POSTGRES:
        cur = conn.cursor(cursor_factory=RealDictCursor)
    else:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
    
    cur.execute(f'SELECT * FROM runs WHERE run_name = {param}', (run_name,))
    run = cur.fetchone()
    
    if not run:
        cur.close()
        conn.close()
        return None
    
    run_id = run['id'] if USE_POSTGRES else run[0]
    
    cur.execute(f'SELECT * FROM pages WHERE run_id = {param} ORDER BY page_number', (run_id,))
    pages = cur.fetchall()
    
    cur.execute(f'SELECT * FROM figures WHERE run_id = {param} ORDER BY page_number', (run_id,))
    figures = cur.fetchall()
    
    cur.execute(f'SELECT * FROM tables WHERE run_id = {param} ORDER BY page_number', (run_id,))
    tables = cur.fetchall()
    
    cur.execute(f'SELECT * FROM sections WHERE run_id = {param}', (run_id,))
    sections = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return {
        'run': dict(run),
        'pages': [dict(p) for p in pages],
        'figures': [dict(f) for f in figures],
        'tables': [dict(t) for t in tables],
        'sections': [dict(s) for s in sections]
    }

def save_summary(section_db_id, summary_type, content, bullets=None):
    """Save an AI-generated summary for a section."""
    conn = get_connection()
    cur = conn.cursor()
    param = '%s' if USE_POSTGRES else '?'
    
    cur.execute(f'''
        INSERT INTO summaries (section_id, summary_type, content, bullets_json)
        VALUES ({param}, {param}, {param}, {param})
    ''', (
        section_db_id,
        summary_type,
        content,
        json.dumps(bullets) if bullets else None
    ))
    
    conn.commit()
    cur.close()
    conn.close()
