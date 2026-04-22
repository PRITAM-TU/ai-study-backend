"""
All LLM prompts used across AI features.
Centralized here for easy modification and consistency.
"""

# ── Summary ──
SUMMARY_SYSTEM = """You are an expert academic summarizer. Create comprehensive, well-structured summaries of study materials. Always respond in valid JSON format."""

SUMMARY_USER = """Summarize the following study material. Provide:
1. A brief overview (2-3 sentences)
2. Key topics covered
3. Main points for each topic
4. Important definitions or formulas
5. A conclusion

Study Material:
{text}

Respond in this exact JSON format:
{{
    "overview": "Brief overview...",
    "key_topics": ["topic1", "topic2", ...],
    "main_points": [
        {{
            "topic": "Topic Name",
            "points": ["point 1", "point 2", ...]
        }}
    ],
    "definitions": [
        {{
            "term": "Term",
            "definition": "Definition..."
        }}
    ],
    "conclusion": "Conclusion..."
}}"""

# ── Quiz ──
QUIZ_SYSTEM = """You are an expert quiz creator for educational content. Create questions that test understanding at various difficulty levels. Always respond in valid JSON format."""

QUIZ_USER = """Create a quiz with {num_questions} questions based on the following study material.
Difficulty level: {difficulty} (easy/medium/hard)

For each question, create a mix of multiple choice and true/false questions.

Study Material:
{text}

Respond in this exact JSON format:
{{
    "quiz_title": "Quiz on [Topic]",
    "difficulty": "{difficulty}",
    "questions": [
        {{
            "id": 1,
            "type": "mcq",
            "question": "Question text?",
            "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
            "correct_answer": "A",
            "explanation": "Why this is correct..."
        }},
        {{
            "id": 2,
            "type": "true_false",
            "question": "Statement to evaluate",
            "correct_answer": "True",
            "explanation": "Why this is true/false..."
        }}
    ]
}}"""

# ── Flashcards ──
FLASHCARDS_SYSTEM = """You are an expert flashcard creator optimized for spaced repetition learning. Create clear, concise Q&A pairs that aid memorization. Always respond in valid JSON format."""

FLASHCARDS_USER = """Create {num_cards} flashcards from the following study material.
Focus on key concepts, definitions, formulas, and important facts.

Study Material:
{text}

Respond in this exact JSON format:
{{
    "topic": "Main Topic",
    "flashcards": [
        {{
            "id": 1,
            "front": "Question or prompt",
            "back": "Answer or explanation",
            "category": "Category/subtopic",
            "difficulty": "easy|medium|hard"
        }}
    ]
}}"""

# ── Exam Mode ──
EXAM_PREDICT_SYSTEM = """You are an expert exam preparation assistant. Analyze study materials to predict likely exam questions and identify important topics. Always respond in valid JSON format."""

EXAM_PREDICT_USER = """Analyze the following study material and predict exam-worthy content.

Study Material:
{text}

Provide:
1. Most important topics (ranked by likelihood of appearing in exam)
2. Predicted exam questions (both MCQ and subjective/essay)
3. Topics that appear repeatedly (high-weight topics)
4. Key formulas or definitions likely to be tested

Respond in this exact JSON format:
{{
    "important_topics": [
        {{
            "rank": 1,
            "topic": "Topic name",
            "importance": "high|medium",
            "reason": "Why this is important"
        }}
    ],
    "predicted_questions": {{
        "mcq": [
            {{
                "id": 1,
                "question": "Question?",
                "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
                "correct_answer": "A",
                "explanation": "Why..."
            }}
        ],
        "subjective": [
            {{
                "id": 1,
                "question": "Essay/short answer question?",
                "key_points": ["Point 1", "Point 2", ...],
                "marks": 5
            }}
        ]
    }},
    "repeated_topics": ["topic1", "topic2", ...],
    "key_formulas": [
        {{
            "name": "Formula name",
            "formula": "The formula",
            "usage": "When to use it"
        }}
    ]
}}"""

# ── Lazy Mode (TTS Script) ──
LAZY_MODE_SYSTEM = """You are an expert educational content narrator. Convert study materials into engaging, conversational audio scripts that are easy to listen to and understand. Write as if you're explaining to a friend."""

LAZY_MODE_USER = """Convert the following study material into a structured audio script for text-to-speech.

Requirements:
- Use conversational, engaging language
- Break into clear sections with natural pauses (use "..." for pauses)
- Add transitions between topics
- Summarize key points at the end of each section
- Make complex concepts easy to understand by ear

Study Material:
{text}

Respond in this exact JSON format:
{{
    "title": "Script title",
    "estimated_duration_minutes": 10,
    "sections": [
        {{
            "title": "Section Title",
            "script": "The narration text for this section...",
            "key_takeaways": ["takeaway 1", "takeaway 2"]
        }}
    ],
    "full_script": "The complete combined script text for TTS..."
}}"""
