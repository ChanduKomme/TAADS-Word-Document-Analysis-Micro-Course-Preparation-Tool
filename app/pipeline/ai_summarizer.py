"""
AI-powered summarization for PDF sections using Ollama.
Generates descriptive section identifiers and learning bullets.
"""

import os
import json
import hashlib
import requests
from typing import List, Dict, Any
from pathlib import Path

# Cache directory for deterministic results - use absolute path relative to this file
CACHE_DIR = Path(__file__).parent / ".ai_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _get_cache_path(cache_type: str, content_hash: str) -> Path:
    """Get cache file path for a given type and content hash."""
    return CACHE_DIR / f"{cache_type}_{content_hash}.json"


def _load_from_cache(cache_type: str, content_hash: str):
    """Load cached result if exists."""
    cache_path = _get_cache_path(cache_type, content_hash)
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_to_cache(cache_type: str, content_hash: str, data):
    """Save result to cache."""
    cache_path = _get_cache_path(cache_type, content_hash)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass  # Cache save failure is non-critical


def initialize_openai_client():
    """
    Stub function for compatibility - not used when running with Ollama.
    Returns None since we're using Ollama instead.
    """
    return None


def generate_section_identifier(title: str, content: str, client=None) -> str:
    """
    Generate a descriptive topic identifier from section content.
    Forwards to Ollama-based generation.
    """
    return generate_section_identifier_ollama(title, content)


def generate_section_identifier_ollama(title: str,
                                       content: str,
                                       ollama_url: str = None) -> str:
    """
    Generate a natural language section identifier using Ollama.
    DETERMINISTIC: Uses file-based cache to ensure same content always returns same identifier.

    Args:
        title: Section title from PDF (used for context, NOT to be repeated)
        content: Section text content (ACTUAL DATA TO ANALYZE)
        ollama_url: Ollama API endpoint (default from env OLLAMA_URL or localhost:11434)

    Returns:
        Natural language identifier (3-6 words capturing the main topic, different from title)
    """
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    # NORMALIZE content before hashing to ensure determinism
    normalized_title = " ".join(title.split())
    normalized_content = " ".join(content.split())[:600]

    # Generate hash for cache lookup
    cache_hash = hashlib.sha256(
        f"identifier:{normalized_title}:{normalized_content}".encode(
        )).hexdigest()[:16]

    # CHECK CACHE FIRST - return cached result if exists
    cached = _load_from_cache("identifier", cache_hash)
    if cached is not None:
        print(f"  [Cache ✓] Returning cached identifier: '{cached}'")
        return cached

    def _try_generate(prompt_text, timeout_sec=60):
        """Helper to generate identifier with retry on timeout"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(f"{ollama_url}/api/generate",
                                         json={
                                             "model": "qwen2.5",
                                             "prompt": prompt_text,
                                             "stream": False,
                                             "temperature": 0.0,
                                             "num_predict": 50
                                         },
                                         timeout=timeout_sec)
                response.raise_for_status()
                data = response.json()
                identifier = data.get("response", "").strip()

                # Clean up response: remove quotes, extra punctuation, newlines
                identifier = identifier.replace('"', '').replace(
                    "'", '').replace(".", "").replace("\n", "").strip()

                # Remove "BAD EXAMPLES" or any explanatory text that Ollama might add
                if "BAD" in identifier or "GOOD" in identifier or "example" in identifier.lower(
                ):
                    identifier = ""

                return identifier if identifier else None
            except requests.exceptions.Timeout:
                print(
                    f"    [Timeout] Attempt {attempt + 1}/{max_retries}, retrying..."
                )
                timeout_sec += 15  # Increase timeout on each retry
                continue
            except Exception as e:
                print(f"    [Error] {e}")
                return None
        return None

    # Build the prompt - handles both educational and metadata content
    prompt = f"""You are analyzing a document section.

Section Title:
{title}

Section Content:
{content[:600]}

Your task: Generate ONE SHORT PHRASE (3-6 words) that describes what this section contains or explains.

For educational content, describe the main topic.
For metadata (names, credits, references), describe what type of information it provides.

Generate ONLY the phrase, no explanation or quotes."""

    try:
        print(f"  [Ollama] '{title[:35]}...'")
        identifier = _try_generate(prompt)

        if identifier and len(identifier) > 2:
            print(f"  [Ollama ✓] Generated: '{identifier}'")
            _save_to_cache("identifier", cache_hash, identifier)
            return identifier

        print(
            f"  [Ollama ⚠] Could not generate unique identifier, using title")
        return title

    except Exception as e:
        print(f"  [Ollama ERROR] {ollama_url}: {e}")
        return title


def generate_learning_bullets_ollama(content: str,
                                     num_bullets: int = 4,
                                     ollama_url: str = None) -> List[str]:
    """
    Generate concise learning bullets from section content using Ollama.
    DETERMINISTIC: Uses file-based cache to ensure same content always returns same bullets.

    Handles both educational content and metadata (names, credits, references).

    Args:
        content: Section text content
        num_bullets: Number of bullets to generate (default 4)
        ollama_url: Ollama API endpoint (default from env OLLAMA_URL or localhost:11434)

    Returns:
        List of learning bullet points (10-15 words each)
    """
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    # Handle very short or empty content
    if not content or len(content.strip()) < 10:
        return ["This section contains minimal content."]

    # NORMALIZE content before hashing to ensure determinism
    normalized_content = " ".join(content.split())[:600]
    word_count = len(normalized_content.split())

    # Generate hash for cache lookup
    content_hash = hashlib.sha256(
        f"bullets:{normalized_content}:{num_bullets}".encode()).hexdigest(
        )[:16]

    # CHECK CACHE FIRST - return cached result if exists
    cached = _load_from_cache("bullets", content_hash)
    if cached is not None:
        print(f"  [Cache ✓] Returning cached bullets")
        return cached

    # Adaptive prompt based on content type
    # Check if content appears to be mostly names/credits
    import re
    name_indicators = [
        'Prof.', 'Dr.', 'Ph.D.', 'Editor', 'Publisher', 'Staff', 'Reviewer'
    ]
    is_metadata = any(indicator in content
                      for indicator in name_indicators) and word_count < 150

    if is_metadata:
        prompt = f"""This section contains metadata like names, credits, or references.
Extract {num_bullets} key points describing what information is provided.
Each point should be 10-15 words.
One point per line, no bullet symbols or numbers.

Text:
{content[:500]}"""
    else:
        prompt = f"""Extract {num_bullets} key learning points from this text.
Each point should be:
- A complete, clear sentence
- 10-15 words long
- Informative and educational

One point per line, no bullet symbols or numbers.

Text:
{content[:500]}"""

    try:
        print(f"  [Ollama] Generating {num_bullets} bullets...")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "qwen2.5",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.0,
                "num_predict": 150
            },
            timeout=90  # Increased timeout for reliability
        )
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "").strip()

        # Parse bullets from response
        bullets = []
        for line in result.split('\n'):
            line = line.strip()
            # Remove common prefixes like "1.", "- ", "* ", etc.
            if line:
                # Remove numbering
                cleaned = re.sub(r'^[\d]+[.\)]\s*', '', line)
                cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
                cleaned = cleaned.strip()
                if cleaned and len(cleaned) > 10:
                    bullets.append(cleaned)

        bullets = bullets[:num_bullets]

        # Fallback if no meaningful bullets were generated
        if not bullets:
            if is_metadata:
                bullets = [
                    f"This section lists {word_count} words of contributor and editorial information.",
                    "Contains names and roles of publishers, editors, and reviewers.",
                    "Provides attribution and credits for the document.",
                    "Lists the editorial team and their affiliations."
                ][:num_bullets]
            else:
                bullets = [
                    f"This section contains {word_count} words of content."
                ]

        # SAVE TO CACHE for future requests
        _save_to_cache("bullets", content_hash, bullets)
        print(f"  [Ollama ✓] Generated {len(bullets)} bullets")

        return bullets

    except requests.exceptions.Timeout:
        print(f"  [Ollama TIMEOUT] Request timed out after 90 seconds")
        # Return meaningful fallback
        if is_metadata:
            return [
                "This section contains contributor and editorial credits.",
                "Lists names of publishers, editors, and reviewers.",
                f"Contains approximately {word_count} words of attribution information."
            ][:num_bullets]
        return [
            f"This section contains {word_count} words. Summary generation timed out."
        ]

    except requests.exceptions.ConnectionError:
        print(f"  [Ollama ERROR] Cannot connect to {ollama_url}")
        return ["Could not connect to Ollama. Please ensure it is running."]

    except Exception as e:
        print(f"  [Ollama ERROR] {e}")
        return [
            f"Summary generation error. Section contains {word_count} words."
        ]


def generate_review_questions_ollama(content: str,
                                      num_questions: int = 2,
                                      ollama_url: str = None) -> List[str]:
    """
    Generate review questions from section content using Ollama.
    Questions directly relate to the extracted text.
    DETERMINISTIC: Uses file-based cache for consistent results.

    Args:
        content: Section text content
        num_questions: Number of questions to generate (default 2)
        ollama_url: Ollama API endpoint

    Returns:
        List of review questions
    """
    import re
    
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    if not content or len(content.strip()) < 50:
        return ["What is the main topic of this section?"]

    normalized_content = " ".join(content.split())[:800]
    word_count = len(normalized_content.split())

    content_hash = hashlib.sha256(
        f"questions:{normalized_content}:{num_questions}".encode()).hexdigest()[:16]

    cached = _load_from_cache("questions", content_hash)
    if cached is not None:
        print(f"  [Cache] Returning cached questions")
        return cached

    prompt = f"""Based on this text, generate exactly {num_questions} review questions.

Requirements:
- Questions must directly test understanding of the specific content
- Each question should be answerable from the text
- Questions should be clear and concise
- One question per line, no numbering

Text:
{content[:600]}

Generate {num_questions} questions:"""

    try:
        print(f"  [Ollama] Generating {num_questions} review questions...")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "qwen2.5",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,
                "num_predict": 150
            },
            timeout=90
        )
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "").strip()

        questions = []
        for line in result.split('\n'):
            line = line.strip()
            if line:
                cleaned = re.sub(r'^[\d]+[.\)]\s*', '', line)
                cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
                cleaned = cleaned.strip()
                if cleaned and len(cleaned) > 15 and '?' in cleaned:
                    questions.append(cleaned)

        questions = questions[:num_questions]

        if not questions:
            questions = [
                f"What are the key points discussed in this section?",
                f"How does this content relate to the overall topic?"
            ][:num_questions]

        _save_to_cache("questions", content_hash, questions)
        print(f"  [Ollama] Generated {len(questions)} questions")
        return questions

    except requests.exceptions.Timeout:
        print(f"  [Ollama TIMEOUT] Questions generation timed out")
        return ["What is the main idea of this section?", "What details support the main topic?"][:num_questions]

    except requests.exceptions.ConnectionError:
        print(f"  [Ollama ERROR] Cannot connect to {ollama_url}")
        return ["Could not connect to AI. Please ensure Ollama is running."]

    except Exception as e:
        print(f"  [Ollama ERROR] {e}")
        return [f"Question generation error for {word_count} word section."]


def generate_answer_ollama(question: str,
                           context: str,
                           ollama_url: str = None) -> str:
    """
    Generate an answer to a review question based on the section content.
    
    Args:
        question: The question to answer
        context: The section text to use as context
        ollama_url: Ollama API endpoint
        
    Returns:
        Answer string
    """
    import re
    
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    if not context or len(context.strip()) < 20:
        return "Insufficient context to answer this question."
    
    normalized_context = " ".join(context.split())[:800]
    
    content_hash = hashlib.sha256(
        f"answer:{question}:{normalized_context[:200]}".encode()).hexdigest()[:16]
    
    cached = _load_from_cache("answer", content_hash)
    if cached is not None:
        return cached
    
    prompt = f"""Based on this text, answer the question concisely in 1-2 sentences.

Text:
{context[:600]}

Question: {question}

Answer:"""
    
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "qwen2.5",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
                "num_predict": 100
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        
        if answer:
            _save_to_cache("answer", content_hash, answer)
            return answer
        return "Could not generate answer."
        
    except Exception as e:
        print(f"  [Ollama ERROR] {e}")
        return "Error generating answer."


def generate_learning_bullets(content: str,
                              num_bullets: int = 4,
                              client=None) -> List[str]:
    """
    Generate concise learning bullets from section content.
    Forwards to Ollama-based generation for compatibility.
    """
    return generate_learning_bullets_ollama(content, num_bullets)


def generate_section_summary(title: str,
                             content: str,
                             num_bullets: int = 4,
                             ollama_url: str = None) -> Dict[str, Any]:
    """
    Generate BOTH identifier and bullets for a section.

    Args:
        title: Section title
        content: Section text
        num_bullets: Number of bullets
        ollama_url: Ollama API endpoint

    Returns:
        Dictionary with 'identifier' and 'bullets'
    """
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    identifier = generate_section_identifier_ollama(title, content, ollama_url)
    bullets = generate_learning_bullets_ollama(content, num_bullets,
                                               ollama_url)

    return {
        "identifier": identifier if identifier else title,
        "bullets": bullets if bullets else []
    }


def enhance_section_with_ai(section: Dict[str, Any],
                            ollama_url: str = None) -> Dict[str, Any]:
    """
    Enhance a section with AI-generated identifier and learning bullets.

    Args:
        section: Section dictionary with title, text, etc.
        ollama_url: Ollama API endpoint

    Returns:
        Enhanced section with AI-generated content
    """
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    title = section.get("title", "")
    text = section.get("raw_text", "") or section.get("text", "")

    # Generate summary (identifier + bullets)
    summary = generate_section_summary(title,
                                       text,
                                       num_bullets=4,
                                       ollama_url=ollama_url)

    # Use "identifier" field to match downstream consumers
    section["identifier"] = summary["identifier"]

    if summary["bullets"]:
        section["learning_text"] = summary["bullets"]

    return section


def expand_or_summarize_content(content: str,
                                target_words: int,
                                ollama_url: str = None) -> str:
    """
    Expand or summarize content to reach target word count using Ollama.

    Args:
        content: Original content
        target_words: Target word count
        ollama_url: Ollama API endpoint

    Returns:
        Content adjusted to target word count
    """
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    current_words = len(content.split())

    if abs(current_words - target_words) < 20:
        return content

    if current_words < target_words:
        action = "expand"
        instruction = f"Expand this text to approximately {target_words} words while preserving all key information and adding relevant context."
    else:
        action = "summarize"
        instruction = f"Summarize this text to approximately {target_words} words while preserving all key information."

    prompt = f"""{instruction}

Text:
{content}

Return only the {action}ed version, nothing else."""

    try:
        response = requests.post(f"{ollama_url}/api/generate",
                                 json={
                                     "model": "qwen2.5",
                                     "prompt": prompt,
                                     "stream": False,
                                     "temperature": 0.3,
                                     "num_predict": int(target_words * 2)
                                 },
                                 timeout=120)
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "").strip()
        return result if result else content
    except Exception as e:
        print(f"Content adjustment failed: {e}")
        return content


import re

def generate_learn_controls(content: str, title: str = "", num_questions: int = 4, ollama_url: str = None) -> List[str]:
    """
    Generate learn control questions (free-text questions) from section content using Ollama.
    DETERMINISTIC: Uses file-based cache to ensure same content always returns same questions.
    
    Args:
        content: Section text content
        title: Section title for context
        num_questions: Number of questions to generate (default 4)
        ollama_url: Ollama API endpoint (default from env OLLAMA_URL or localhost:11434)
    
    Returns:
        List of free-text questions for learning assessment
    """
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    # Handle very short or empty content
    if not content or len(content.strip()) < 30:
        return ["What is the main purpose of this section?"]
    
    # NORMALIZE content before hashing to ensure determinism
    normalized_content = " ".join(content.split())[:800]
    normalized_title = " ".join(title.split()) if title else ""
    word_count = len(normalized_content.split())
    
    # Generate hash for cache lookup
    content_hash = hashlib.sha256(f"questions:{normalized_title}:{normalized_content}:{num_questions}".encode()).hexdigest()[:16]
    
    # CHECK CACHE FIRST - return cached result if exists
    cached = _load_from_cache("questions", content_hash)
    if cached is not None:
        print(f"  [Cache] Returning cached questions")
        return cached
    
    # Check if content is metadata (names, credits)
    name_indicators = ['Prof.', 'Dr.', 'Ph.D.', 'Editor', 'Publisher', 'Staff', 'Reviewer']
    is_metadata = any(indicator in content for indicator in name_indicators) and word_count < 150
    
    if is_metadata:
        # For metadata sections, generate simple questions about the document structure
        questions = [
            "Who are the main contributors to this publication?",
            "What roles are mentioned in the editorial team?",
            "How is the editorial board structured?",
            f"What can you learn about the publication from this {title.lower() if title else 'section'}?"
        ][:num_questions]
        _save_to_cache("questions", content_hash, questions)
        return questions
    
    # Build the prompt for educational content
    prompt = f"""You are an educational content expert. Based on the following text, generate {num_questions} thoughtful free-text questions that test understanding of the key concepts.

Section Title: {title if title else "Untitled Section"}

Content:
{content[:700]}

Requirements for questions:
- Questions should test comprehension and critical thinking
- Each question should be answerable from the content
- Questions should be clear and specific (not yes/no questions)
- Questions should encourage reflection and deeper understanding
- Start each question with words like: What, How, Why, Explain, Describe, Compare

Generate exactly {num_questions} questions, one per line, no numbering or bullet points.
Only output the questions, nothing else."""

    try:
        print(f"  [Ollama] Generating {num_questions} learn control questions...")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "qwen2.5",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,
                "num_predict": 250
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "").strip()
        
        # Parse questions from response
        questions = []
        for line in result.split('\n'):
            line = line.strip()
            if line:
                # Remove numbering like "1.", "1)", "Q1:", etc.
                cleaned = re.sub(r'^[\d]+[.\)]\s*', '', line)
                cleaned = re.sub(r'^Q[\d]*[.:]\s*', '', cleaned, flags=re.I)
                cleaned = re.sub(r'^[-*]\s*', '', cleaned)
                cleaned = re.sub(r'^\*\*', '', cleaned)  # Remove markdown bold
                cleaned = re.sub(r'\*\*$', '', cleaned)
                cleaned = cleaned.strip()
                
                # Only keep lines that look like questions
                if cleaned and len(cleaned) > 15:
                    # Ensure it ends with a question mark
                    if not cleaned.endswith('?'):
                        cleaned = cleaned.rstrip('.') + '?'
                    questions.append(cleaned)
        
        questions = questions[:num_questions]
        
        # Fallback if no meaningful questions were generated
        if not questions or len(questions) < 2:
            questions = [
                f"What are the main concepts discussed in this section about {title if title else 'this topic'}?",
                "How do the ideas presented in this section relate to the broader context?",
                "What evidence or examples support the main arguments in this section?",
                "Why is understanding this content important for the overall subject?"
            ][:num_questions]
        
        # SAVE TO CACHE for future requests
        _save_to_cache("questions", content_hash, questions)
        print(f"  [Ollama] Generated {len(questions)} questions")
        
        return questions
        
    except requests.exceptions.Timeout:
        print(f"  [Ollama TIMEOUT] Question generation timed out")
        return [
            f"What are the key points discussed in {title if title else 'this section'}?",
            "How would you summarize the main argument of this section?",
            "What questions do you have after reading this content?"
        ][:num_questions]
        
    except requests.exceptions.ConnectionError:
        print(f"  [Ollama ERROR] Cannot connect to {ollama_url}")
        return ["Could not generate questions. Please ensure Ollama is running."]
        
    except Exception as e:
        print(f"  [Ollama ERROR] {e}")
        return [f"What did you learn from this section about {title if title else 'the topic'}?"]


def generate_answer_for_question(question: str, content: str, title: str = "", ollama_url: str = None) -> str:
    """
    Generate an answer for a review question based on the section content.
    
    Args:
        question: The review question to answer
        content: Section text content to base the answer on
        title: Section title for context
        ollama_url: Ollama API endpoint
    
    Returns:
        Answer string
    """
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    # Use whatever content is available, even if short
    if not content:
        content = title if title else "No content available"
    
    # Generate hash for cache lookup
    normalized_q = " ".join(question.split())
    normalized_content = " ".join(content.split())[:600]
    content_hash = hashlib.sha256(f"answer:{normalized_q}:{normalized_content}".encode()).hexdigest()[:16]
    
    # CHECK CACHE FIRST
    cached = _load_from_cache("answers", content_hash)
    if cached is not None:
        print(f"  [Cache] Returning cached answer")
        return cached
    
    prompt = f"""You are an educational expert. Based on the following section content, provide a clear and concise answer to the question.

Section Title: {title if title else "Untitled Section"}

Content:
{content[:600]}

Question: {question}

Provide a clear, factual answer in 2-3 sentences based ONLY on the content above. Be direct and informative."""

    try:
        print(f"  [Ollama] Generating answer...")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "qwen2.5",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,
                "num_predict": 150
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        
        if not answer:
            answer = "Could not generate an answer. Please review the section content."
        
        # Save to cache
        _save_to_cache("answers", content_hash, answer)
        print(f"  [Ollama] Generated answer")
        
        return answer
        
    except Exception as e:
        print(f"  [Ollama ERROR] {e}")
        return "Could not generate answer. Please ensure Ollama is running."


def generate_qa_pairs(content: str, title: str = "", num_questions: int = 4, ollama_url: str = None) -> List[Dict[str, str]]:
    """
    Generate question-answer pairs for a section.
    
    Args:
        content: Section text content
        title: Section title for context
        num_questions: Number of Q&A pairs to generate
        ollama_url: Ollama API endpoint
    
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    if not content or len(content.strip()) < 30:
        return [{"question": "What is the main purpose of this section?", "answer": "The section content is too short to determine."}]
    
    # Generate hash for cache lookup
    normalized_content = " ".join(content.split())[:800]
    content_hash = hashlib.sha256(f"qa_pairs:{title}:{normalized_content}:{num_questions}".encode()).hexdigest()[:16]
    
    # CHECK CACHE FIRST
    cached = _load_from_cache("qa_pairs", content_hash)
    if cached is not None:
        print(f"  [Cache] Returning cached Q&A pairs")
        return cached
    
    prompt = f"""You are an educational expert. Based on the following text, generate {num_questions} question-answer pairs that test understanding of the key concepts.

Section Title: {title if title else "Untitled Section"}

Content:
{content[:700]}

For each pair:
- Question: A clear, thoughtful question testing comprehension
- Answer: A concise 1-2 sentence answer based on the content

Format your response EXACTLY like this:
Q: [question]
A: [answer]

Q: [question]
A: [answer]

Generate exactly {num_questions} Q&A pairs."""

    try:
        print(f"  [Ollama] Generating {num_questions} Q&A pairs...")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "qwen2.5",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,
                "num_predict": 500
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "").strip()
        
        # Parse Q&A pairs
        qa_pairs = []
        lines = result.split('\n')
        current_q = None
        current_a = None
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('Q:') or line.upper().startswith('Q '):
                if current_q and current_a:
                    qa_pairs.append({"question": current_q, "answer": current_a})
                current_q = re.sub(r'^Q[:\s]+', '', line, flags=re.I).strip()
                current_a = None
            elif line.upper().startswith('A:') or line.upper().startswith('A '):
                current_a = re.sub(r'^A[:\s]+', '', line, flags=re.I).strip()
        
        # Don't forget the last pair
        if current_q and current_a:
            qa_pairs.append({"question": current_q, "answer": current_a})
        
        qa_pairs = qa_pairs[:num_questions]
        
        # Fallback if parsing failed
        if not qa_pairs:
            qa_pairs = [
                {"question": f"What are the main concepts discussed in {title if title else 'this section'}?", 
                 "answer": "Please review the section content for the main concepts."},
                {"question": "How do the ideas presented relate to the broader context?",
                 "answer": "The ideas connect to the overall theme of the document."}
            ][:num_questions]
        
        # Save to cache
        _save_to_cache("qa_pairs", content_hash, qa_pairs)
        print(f"  [Ollama] Generated {len(qa_pairs)} Q&A pairs")
        
        return qa_pairs
        
    except Exception as e:
        print(f"  [Ollama ERROR] {e}")
        return [{"question": "Could not generate questions.", "answer": "Please ensure Ollama is running."}]


def generate_section_with_learn_controls(title: str, content: str, num_bullets: int = 4, num_questions: int = 4, ollama_url: str = None) -> Dict[str, Any]:
    """
    Generate complete section enhancement: identifier, bullets, AND learn control questions.
    
    Args:
        title: Section title
        content: Section text
        num_bullets: Number of bullet points
        num_questions: Number of learn control questions
        ollama_url: Ollama API endpoint
    
    Returns:
        Dictionary with 'identifier', 'bullets', and 'questions'
    """
    if ollama_url is None:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    identifier = generate_section_identifier_ollama(title, content, ollama_url)
    bullets = generate_learning_bullets_ollama(content, num_bullets, ollama_url)
    questions = generate_learn_controls(content, title, num_questions, ollama_url)
    
    return {
        "identifier": identifier if identifier else title,
        "bullets": bullets if bullets else [],
        "questions": questions if questions else []
    }
