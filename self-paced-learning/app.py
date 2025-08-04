import os
import json
import shutil
import re  # For parsing AI responses
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    session,
    redirect,
    url_for,
)  # Added redirect, url_for
from openai import OpenAI
from dotenv import load_dotenv
from utils.data_loader import DataLoader

#  Load Environment Variables
load_dotenv()

#  App Configuration
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_KEY")
if not app.secret_key:
    app.logger.warning(
        "FLASK_KEY not set, using a default secret key. Please set this in your .env file for production."
    )
    app.secret_key = (
        "your_default_secret_key_for_development_12345_v2"  # Fallback for local dev
    )

#  OpenAI Client Initialization
# Ensure OPEN_API_KEY is set in your .env file
try:
    api_key = os.getenv("OPEN_API_KEY")
    if not api_key:
        raise ValueError("OPEN_API_KEY not found in environment variables.")
    # Simple initialization without extra parameters
    client = OpenAI(api_key=api_key)
    # Test the client
    app.logger.info("OpenAI client initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None  # Allow app to run but AI features will fail if client is None

#  Constants and Global Settings
MASTERY_THRESHOLD = 0.80  # 80% score to consider targeted weak topics mastered

#  Initialize DataLoader
DATA_ROOT_PATH = os.path.join(os.path.dirname(__file__), "data")
data_loader = DataLoader(DATA_ROOT_PATH)


#  Helper Functions
def get_session_key(subject: str, subtopic: str, key_type: str) -> str:
    """Generate session key with subject/subtopic prefix."""
    return f"{subject}_{subtopic}_{key_type}"


def get_subject_keywords(subject: str) -> list:
    """Get allowed AI analysis keywords for a subject."""
    return data_loader.get_subject_keywords(subject)


def get_lessons_by_tags(subject: str, tags: list) -> list:
    """Get lessons that match the given tags."""
    return data_loader.find_lessons_by_tags(subject, tags)


def get_quiz_data(subject: str, subtopic: str) -> list:
    """Get quiz questions for a subject/subtopic."""
    return data_loader.get_quiz_questions(subject, subtopic)


def check_prerequisites(
    subject: str, subtopic: str, user_progress: dict = None
) -> tuple[bool, list]:
    """Check if user has completed all prerequisites for a subtopic.

    Args:
        subject: Subject ID
        subtopic: Subtopic ID
        user_progress: User's completion progress (for future implementation)

    Returns:
        tuple: (prerequisites_met, missing_prerequisites)
    """
    try:
        subject_config = data_loader.load_subject_config(subject)
        if not subject_config:
            return True, []

        subtopic_data = subject_config.get("subtopics", {}).get(subtopic, {})
        prerequisites = subtopic_data.get("prerequisites", [])

        if not prerequisites:
            return True, []

        # For now, we'll assume no progress tracking is implemented
        # In the future, this would check actual user completion data
        # For testing purposes, we'll return that prerequisites are not met
        return False, prerequisites

    except Exception as e:
        app.logger.error(f"Error checking prerequisites for {subject}/{subtopic}: {e}")
        return True, []  # Default to allowing access if there's an error


def is_admin_override_active(session_data: dict) -> bool:
    """Check if admin override is active for current session."""
    return session_data.get("admin_override", False)


def get_question_pool(subject: str, subtopic: str) -> list:
    """Get question pool for remedial quizzes."""
    return data_loader.get_question_pool_questions(subject, subtopic)


def get_lesson_plans(subject: str, subtopic: str) -> dict:
    """Get lesson plans for a subject/subtopic."""
    lessons_data = data_loader.load_lesson_plans(subject, subtopic)
    return lessons_data.get("lessons", {}) if lessons_data else {}


def get_video_data(subject: str, subtopic: str) -> dict:
    """Get video data for a subject/subtopic."""
    videos_data = data_loader.load_videos(subject, subtopic)
    return videos_data.get("videos", {}) if videos_data else {}


def format_quiz_bank_for_ai_prompt(quiz_bank, title="Reference Quiz Bank"):
    """Formats a quiz bank (like FUNCTIONS_QUIZ) into a string for AI prompts."""
    if not quiz_bank:  # Handles empty or None quiz_bank
        return f"\n--- {title}: Not available or empty. ---\n"
    text = f"\n--- {title} ---\n"
    for i, q_data in enumerate(quiz_bank):
        text += f"Q{i+1}: {q_data.get('question', 'N/A')}\n"
        text += f"Options: {json.dumps(q_data.get('options', []))}\n"
        answer_idx = q_data.get("answer_index")
        options = q_data.get("options", [])
        if (
            answer_idx is not None
            and isinstance(answer_idx, int)
            and 0 <= answer_idx < len(options)
        ):
            text += f"Correct Answer: {options[answer_idx]}\n\n"
        else:
            text += f"Correct Answer: Not specified or invalid index (index: {answer_idx}, options_len: {len(options)})\n\n"
    text += "--- End of Bank ---\n"
    return text


def parse_ai_json_from_text(ai_response_string, expected_type_is_list=True):
    """
    Attempts to parse a JSON object or list from a string that might contain other text,
    including markdown code blocks.
    """
    if not ai_response_string:
        app.logger.warning("AI response string is empty in parse_ai_json_from_text.")
        return None

    # Pattern to extract JSON from ```json ... ``` or raw {...} / [...]
    if expected_type_is_list:
        pattern = r"```json\s*(\[[\s\S]*?\])\s*```|(\[[\s\S]*?\](?!\s*:))"
    else:  # expecting an object
        pattern = r"```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?\}(?!\s*:))"

    match = re.search(pattern, ai_response_string, re.DOTALL)

    json_str = None
    if match:
        # Try groups in order of preference (group 1 for markdown, group 2 for raw)
        for group_content in match.groups():
            if group_content:
                json_str = group_content
                break

    if json_str:
        try:
            json_str_cleaned = json_str.strip()
            parsed_json = json.loads(json_str_cleaned)

            if expected_type_is_list and not isinstance(parsed_json, list):
                app.logger.warning(
                    f"AI returned JSON but not the expected list type. Got: {type(parsed_json)}. From: {json_str_cleaned[:100]}"
                )
                return None
            if not expected_type_is_list and not isinstance(parsed_json, dict):
                app.logger.warning(
                    f"AI returned JSON but not the expected dict type. Got: {type(parsed_json)}. From: {json_str_cleaned[:100]}"
                )
                return None
            return parsed_json
        except json.JSONDecodeError as e:
            app.logger.error(
                f"JSONDecodeError in parse_ai_json_from_text: {e}. Attempted to parse: {json_str_cleaned[:200]}"
            )
            return None

    app.logger.warning(
        f"Could not find or parse expected JSON structure in AI response: {ai_response_string[:500]}..."
    )
    return None


def call_openai_api(
    prompt_text,
    system_message="",
    model="gpt-4-0613",
    max_tokens=1500,
    expect_json_output=False,
):
    """
    Clean helper function to call OpenAI API with proper error handling.

    Args:
        prompt_text: The user prompt/question
        system_message: System instructions for the AI
        model: OpenAI model to use (default: gpt-4o)
        max_tokens: Maximum tokens for response
        expect_json_output: Whether to request JSON format response

    Returns:
        str: AI response content or None if failed
    """

    # Validate client
    if not client:
        app.logger.error("OpenAI client not initialized.")
        return None

    try:

        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt_text})

        # Build request args
        completion_args = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Add JSON mode for supported models (gpt-4o only)
        if expect_json_output and "gpt-4o" in model:
            completion_args["response_format"] = {"type": "json_object"}

        # Make API call
        app.logger.info(f"Making OpenAI API call with model: {model}")
        response = client.chat.completions.create(**completion_args)

        content = response.choices[0].message.content.strip()
        app.logger.info(
            f"OpenAI API call successful. Response length: {len(content)} chars"
        )

        return content

    except Exception as e:
        app.logger.error(f"OpenAI API call failed: {e}")
        return None


#  Main Application Routes
@app.route("/")
def subject_selection():
    """New home page showing all available subjects."""
    try:
        # Auto-discover subjects from individual subject_info.json files
        subjects = data_loader.discover_subjects()
        
        # Fallback to default Python subject if no subjects found
        if not subjects:
            subjects = {
                "python": {
                    "name": "Python Programming",
                    "description": "Master Python from basics to advanced topics including functions, loops, data structures, and more.",
                    "icon": "fab fa-python",
                    "color": "#3776ab",
                    "status": "active",
                    "subtopic_count": 6,
                }
            }

        return render_template("subject_selection.html", subjects=subjects)
    except Exception as e:
        app.logger.error(f"Error loading subject selection: {e}")
        # Fallback to legacy index if there's an error
        return redirect(url_for("python_subject_page"))


@app.route("/subjects/<subject>")
def subject_page(subject):
    """Display subtopics for a specific subject."""
    try:
        # Load subject configuration
        subject_config = data_loader.load_subject_config(subject)
        if not subject_config:
            app.logger.error(f"Subject config not found for: {subject}")
            return redirect(url_for("subject_selection"))

        subject_info = subject_config.get("subject_info", {})
        subtopics = subject_config.get("subtopics", {})

        # Update question counts dynamically by checking for quiz data
        for subtopic_id, subtopic_data in subtopics.items():
            quiz_data = data_loader.load_quiz_data(subject, subtopic_id)
            if quiz_data and quiz_data.get("questions"):
                # Count questions in the initial quiz
                question_count = len(quiz_data.get("questions", []))
                subtopic_data["question_count"] = question_count
            else:
                # No quiz data or empty quiz
                subtopic_data["question_count"] = 0

            # Update video counts dynamically by checking for video data
            try:
                video_data = get_video_data(subject, subtopic_id)
                video_count = len(video_data) if video_data else 0
                subtopic_data["video_count"] = video_count
            except Exception as e:
                app.logger.debug(
                    f"No video data found for {subject}/{subtopic_id}: {e}"
                )
                subtopic_data["video_count"] = 0

        # Sort subtopics by order
        sorted_subtopics = dict(
            sorted(subtopics.items(), key=lambda x: x[1].get("order", 999))
        )

        return render_template(
            "python_subject.html",
            subject=subject,
            subject_info=subject_info,
            subtopics=sorted_subtopics,
        )
    except Exception as e:
        app.logger.error(f"Error loading subject page for {subject}: {e}")
        return redirect(url_for("subject_selection"))


@app.route("/legacy")
def legacy_index():
    """Legacy route - redirects to Python subject page."""
    return redirect(url_for("subject_page", subject="python"))


@app.route("/python")
def python_subject_page():
    """Direct route to Python subject - for backward compatibility."""
    return redirect(url_for("subject_page", subject="python"))


@app.route("/api/video/<topic_key>")
def get_video_api_legacy(topic_key):
    """Legacy video API route for backward compatibility."""
    VIDEO_DATA = {
        "loops": {
            "title": "Python Loops: For and While",
            "url": "https://www.youtube.com/watch?v=94UHCEmprCY",
            "description": "Learn how to automate repetitive tasks using for and while loops, understand iterables, and control flow statements.",
        },
        "functions": {
            "title": "Python Functions Masterclass",
            "url": "https://www.youtube.com/embed/94UHCEmprCY?enablejsapi=1",
            "description": "Master Python functions, parameters, return values, and scope. Learn how to write reusable code blocks.",
        },
        "arrays": {
            "title": "Understanding NumPy Arrays",
            "url": "#",
            "description": "Discover how to use NumPy arrays for efficient numerical computations and data processing in Python.",
        },
        "lists": {
            "title": "Python Lists and List Comprehensions",
            "url": "#",
            "description": "Understand Python's built-in list data structure, methods, and operations for storing collections of items.",
        },
        "sets": {
            "title": "Working with Python Sets",
            "url": "#",
            "description": "Learn about Python's unordered collection of unique elements and set operations like union and intersection.",
        },
        "dictionaries": {
            "title": "Python Dictionaries and Dictionary Comprehensions",
            "url": "#",
            "description": "Explore key-value mappings in Python dictionaries, methods for accessing, modifying, and iterating through data.",
        },
    }
    if topic_key in VIDEO_DATA:
        return jsonify(VIDEO_DATA[topic_key])
    return jsonify({"error": "Topic not found"}), 404


@app.route("/api/video/<subject>/<subtopic>/<topic_key>")
def get_video_api(subject, subtopic, topic_key):
    """Get video data for a specific subject/subtopic/topic."""
    video_data = get_video_data(subject, subtopic)
    if topic_key in video_data:
        return jsonify(video_data[topic_key])
    return jsonify({"error": "Video not found"}), 404


@app.route("/api/progress/update", methods=["POST"])
def update_progress_api():
    data = request.json
    topic_key = data.get("topic")
    progress = data.get("progress")
    if not topic_key or progress is None:
        return jsonify({"error": "Missing data"}), 400
    user_progress = session.get("progress", {})
    user_progress[topic_key] = progress
    session["progress"] = user_progress
    return jsonify({"success": True, "progress": user_progress})


@app.route("/api/progress")
def get_all_progress_api():
    user_progress = session.get("progress", {})
    return jsonify(user_progress)


@app.route("/api/admin/status")
def get_admin_status():
    """Check if current user has admin privileges."""
    is_admin = is_admin_override_active(session)
    return jsonify({"is_admin": is_admin})


@app.route("/api/admin/mark_complete", methods=["POST"])
def admin_mark_complete():
    """Admin endpoint to mark topics as complete."""
    if not is_admin_override_active(session):
        return jsonify({"success": False, "error": "Admin privileges required"}), 403

    data = request.get_json()
    topic = data.get("topic")

    if not topic:
        return jsonify({"success": False, "error": "Topic is required"}), 400

    # Get current progress
    user_progress = session.get("progress", {})

    # Mark topic as 100% complete
    user_progress[topic] = 100
    session["progress"] = user_progress

    return jsonify({"success": True, "topic": topic, "progress": 100})


@app.route("/quiz/<subject>/<subtopic>")
def quiz_page(subject, subtopic):
    """Serves the initial quiz for any subject/subtopic."""
    # Validate that the subject/subtopic exists
    if not data_loader.validate_subject_subtopic(subject, subtopic):
        return f"Error: Subject '{subject}' with subtopic '{subtopic}' not found.", 404

    # Check prerequisites unless admin override is active
    admin_override = is_admin_override_active(session)
    if not admin_override:
        prerequisites_met, missing_prerequisites = check_prerequisites(
            subject, subtopic
        )
        if not prerequisites_met:
            # Get subject config for display names
            subject_config = data_loader.load_subject_config(subject)
            missing_names = []
            if subject_config and "subtopics" in subject_config:
                for req in missing_prerequisites:
                    req_data = subject_config["subtopics"].get(req, {})
                    req_name = req_data.get("name", req.replace("-", " ").title())
                    missing_names.append(req_name)

            return render_template(
                "prerequisites_error.html",
                subject=subject,
                subtopic=subtopic,
                missing_prerequisites=missing_names,
                missing_ids=missing_prerequisites,
            )

    # Clear previous session data for this subject/subtopic
    session_prefix = f"{subject}_{subtopic}"
    keys_to_remove = [key for key in session.keys() if key.startswith(session_prefix)]
    for key in keys_to_remove:
        session.pop(key, None)

    # Load quiz data
    quiz_questions = get_quiz_data(subject, subtopic)
    quiz_title = data_loader.get_quiz_title(subject, subtopic)

    if not quiz_questions:
        return f"Error: No quiz questions found for {subject}/{subtopic}.", 404

    # Set session data with prefixed keys
    session[get_session_key(subject, subtopic, "current_quiz_type")] = "initial"
    session[get_session_key(subject, subtopic, "questions_served_for_analysis")] = (
        quiz_questions
    )
    session["current_subject"] = subject
    session["current_subtopic"] = subtopic

    return render_template(
        "quiz.html",
        questions=quiz_questions,
        quiz_title=quiz_title,
        admin_override=admin_override,
    )


# Legacy route for backward compatibility
@app.route("/quiz/functions")
def quiz_functions_page():
    """Legacy route - redirects to the new structure."""
    return redirect(url_for("quiz_page", subject="python", subtopic="functions"))


@app.route("/analyze", methods=["POST"])
def analyze_quiz():
    user_submitted_answers = request.json.get("answers", {})

    # Get current subject/subtopic from session
    current_subject = session.get("current_subject")
    current_subtopic = session.get("current_subtopic")

    if not current_subject or not current_subtopic:
        app.logger.error("No current subject/subtopic found in session for analysis.")
        return (
            jsonify(
                {"feedback": "Error: Quiz session data not found.", "weak_topics": []}
            ),
            400,
        )

    # Get questions that were served for analysis using prefixed session key
    questions_for_analysis = session.get(
        get_session_key(
            current_subject, current_subtopic, "questions_served_for_analysis"
        ),
        [],
    )

    if not questions_for_analysis:
        app.logger.error("No questions found in session for analysis.")
        return (
            jsonify(
                {"feedback": "Error: Quiz session data not found.", "weak_topics": []}
            ),
            400,
        )

    submission_details_list = []
    correct_answers = 0
    total_questions = len(questions_for_analysis)

    for i, q_data in enumerate(questions_for_analysis):
        user_answer = user_submitted_answers.get(f"q{i}", "[No answer provided]")
        question_type = q_data.get(
            "type", "multiple_choice"
        )  # Default to multiple_choice for backward compatibility
        status = "Incorrect"  # Default status

        detail = (
            f"Question {i+1} (Type: {question_type}): {q_data.get('question', 'N/A')}\n"
        )
        detail += f"Student's Answer:\n---\n{user_answer}\n---\n"

        #  Grading Logic
        if question_type == "multiple_choice":
            correct_answer_index = q_data.get("answer_index")
            options = q_data.get("options", [])
            if 0 <= correct_answer_index < len(options):
                correct_answer_text = options[correct_answer_index]
                if user_answer == correct_answer_text:
                    status = "Correct"
                    correct_answers += 1
                else:
                    detail += f"Correct Answer: {correct_answer_text}\n"
            else:
                status = "Invalid Question Data"

        elif question_type == "fill_in_the_blank":
            correct_answer_text = q_data.get("correct_answer", "")
            # Make comparison case-insensitive and trim whitespace
            # Allow for multiple correct answers separated by commas
            correct_answers_list = [
                ans.strip().lower() for ans in correct_answer_text.split(",")
            ]
            user_answer_clean = user_answer.strip().lower()
            if user_answer_clean in correct_answers_list:
                status = "Correct"
                correct_answers += 1
            else:
                detail += f"Correct Answer(s): {correct_answer_text}\n"

        elif question_type == "coding":
            # For coding questions, we don't grade automatically.
            # We will let the AI review the code.
            status = "For AI Review"
            # Provide the sample solution for the AI's reference.
            sample_solution = q_data.get("sample_solution", "")
            if sample_solution:
                detail += f"Sample Solution:\n---\n{sample_solution}\n---\n"

        detail += f"Status: {status}\n\n"
        submission_details_list.append(detail)

    full_submission_text = "".join(submission_details_list)

    #  system message to include code evaluation
    system_message = (
        "You are an expert instructor. Your task is to analyze a student's quiz performance, "
        "classify their errors against a predefined list of topics, and evaluate their submitted code. "
        "For 'coding' questions, determine if the student's code correctly solves the problem."
    )

    # Get allowed keywords for the current subject
    allowed_topic_keywords = get_subject_keywords(current_subject)
    allowed_keywords_str = json.dumps(allowed_topic_keywords)

    # DEBUG: Log available keywords
    app.logger.info(
        f"[DEBUG] Available keywords for {current_subject}: {allowed_topic_keywords}"
    )
    app.logger.info(f"[DEBUG] Total keywords available: {len(allowed_topic_keywords)}")

    prompt = (
        "You are analyzing a student's quiz submission which includes multiple choice, fill-in-the-blank, and coding questions.\n"
        "Based on the incorrect answers and their submitted code, identify the concepts they are weak in.\n"
        f"You **MUST** choose the weak concepts from this predefined list ONLY: {allowed_keywords_str}\n\n"
        "For coding questions marked 'For AI Review', evaluate if the student's code:\n"
        "1. Correctly solves the problem\n"
        "2. Uses appropriate syntax and conventions\n"
        "3. Demonstrates understanding of the underlying concepts\n"
        "Compare their code with the provided sample solution.\n\n"
        "Provide your analysis as a single JSON object with two keys:\n"
        ' - "detailed_feedback": (string) Your textual analysis, including specific feedback on coding attempts, what they did well, and areas for improvement.\n'
        ' - "weak_concept_tags": (JSON list of strings) The list of weak concepts from the ALLOWED KEYWORDS list. If there are no weaknesses, provide an empty list `[]`.\n\n'
        "Here is the student's submission:\n"
        "--- START OF SUBMISSION ---\n"
        f"{full_submission_text}"
        "--- END OF SUBMISSION ---\n"
    )

    ai_response_content = call_openai_api(
        prompt,
        system_message,
        model="gpt-4-0613",
        max_tokens=1500,  # Increased tokens for more detailed feedback
        expect_json_output=True,
    )

    # DEBUG: Log AI response
    app.logger.info(f"[DEBUG] AI response content: {ai_response_content}")

    if not ai_response_content:
        return (
            jsonify(
                {
                    "feedback": "Error: Could not get analysis from AI.",
                    "weak_topics": [],
                }
            ),
            500,
        )

    json_match = re.search(r"\{[\s\S]*\}", ai_response_content)
    if not json_match:
        app.logger.error(
            f"Could not find JSON in AI response.\nResponse was: {ai_response_content}"
        )
        return (
            jsonify(
                {
                    "feedback": "Error: The analysis response did not contain a valid JSON object.",
                    "weak_topics": [],
                }
            ),
            500,
        )

    try:
        parsed_ai_response = json.loads(json_match.group(0))
        feedback = parsed_ai_response.get(
            "detailed_feedback", "No detailed feedback provided."
        )
        weak_topics = parsed_ai_response.get("weak_concept_tags", [])

        # DEBUG: Log AI's chosen keywords before validation
        app.logger.info(f"[DEBUG] AI chose these keywords: {weak_topics}")

        validated_weak_topics = [
            topic for topic in weak_topics if topic in allowed_topic_keywords
        ]

        # DEBUG: Log validation results
        app.logger.info(f"[DEBUG] Validated keywords: {validated_weak_topics}")
        if len(weak_topics) != len(validated_weak_topics):
            rejected_topics = [
                topic for topic in weak_topics if topic not in allowed_topic_keywords
            ]
            app.logger.warning(
                f"[DEBUG] Rejected keywords (not in allowed list): {rejected_topics}"
            )

        # Store weak topics with subject/subtopic prefix
        session[get_session_key(current_subject, current_subtopic, "weak_topics")] = (
            validated_weak_topics
        )
        app.logger.info(
            f"AI identified weak topics for {current_subject}/{current_subtopic}: {validated_weak_topics}"
        )

        # Also find and store recommended lessons immediately after analysis
        if validated_weak_topics:
            try:
                app.logger.info(
                    f"[DEBUG] Searching for lessons across ALL subtopics in subject '{current_subject}' for weak topics: {validated_weak_topics}"
                )
                matching_lessons = get_lessons_by_tags(
                    current_subject, validated_weak_topics
                )
                session[
                    get_session_key(
                        current_subject, current_subtopic, "recommended_lessons"
                    )
                ] = matching_lessons
                app.logger.info(
                    f"[DEBUG] Found and stored {len(matching_lessons)} recommended lessons for weak topics: {validated_weak_topics}"
                )

                # Detailed logging of found lessons
                if matching_lessons:
                    for lesson in matching_lessons:
                        lesson_subject = lesson.get("subject", "Unknown")
                        lesson_subtopic = lesson.get("subtopic", "Unknown")
                        lesson_title = lesson.get("title", "No title")
                        lesson_tags = lesson.get("tags", [])
                        matching_tags = lesson.get("matching_tags", [])
                        app.logger.info(
                            f"[DEBUG] - Found lesson: '{lesson_title}' in {lesson_subject}/{lesson_subtopic}"
                        )
                        app.logger.info(f"[DEBUG]   - All lesson tags: {lesson_tags}")
                        app.logger.info(
                            f"[DEBUG]   - Matching weak topic tags: {matching_tags}"
                        )
                else:
                    app.logger.warning(
                        f"[DEBUG] No lessons found for any weak topics: {validated_weak_topics}"
                    )
                    # Let's also check what lessons exist in the current subject
                    try:
                        all_current_lessons = get_lesson_plans(
                            current_subject, current_subtopic
                        )
                        app.logger.info(
                            f"[DEBUG] Available lessons in current subtopic {current_subject}/{current_subtopic}: {list(all_current_lessons.keys())}"
                        )
                        for lesson_id, lesson_data in all_current_lessons.items():
                            lesson_tags = lesson_data.get("tags", [])
                            app.logger.info(
                                f"[DEBUG]   - Lesson '{lesson_id}' has tags: {lesson_tags}"
                            )
                    except Exception as e2:
                        app.logger.error(
                            f"[DEBUG] Error checking available lessons: {e2}"
                        )

            except Exception as e:
                app.logger.error(
                    f"Error finding lessons for weak topics {validated_weak_topics}: {e}"
                )
                # Even if lesson search fails, we should log what lessons exist
                try:
                    all_current_lessons = get_lesson_plans(
                        current_subject, current_subtopic
                    )
                    app.logger.info(
                        f"[DEBUG] Fallback: Available lessons in {current_subject}/{current_subtopic}: {list(all_current_lessons.keys())}"
                    )
                except Exception as e2:
                    app.logger.error(f"[DEBUG] Error in fallback lesson check: {e2}")

        # Calculate score percentage
        score_percentage = (
            round((correct_answers / total_questions) * 100)
            if total_questions > 0
            else 0
        )

        return jsonify(
            {
                "feedback": feedback,
                "weak_topics": validated_weak_topics,
                "score": {
                    "correct": correct_answers,
                    "total": total_questions,
                    "percentage": score_percentage,
                },
            }
        )

    except json.JSONDecodeError as e:
        app.logger.error(
            f"Failed to parse extracted AI JSON response: {e}\nExtracted text was: {json_match.group(0)}"
        )
        return (
            jsonify(
                {
                    "feedback": "Error: The analysis response format was invalid.",
                    "weak_topics": [],
                }
            ),
            500,
        )


# Video recommendation removed - videos are tied to lessons, only recommend lessons by tags


@app.route("/generate_remedial_quiz", methods=["GET"])
def generate_remedial_quiz():
    """
    Selects questions from the human-made question pool based on the
    weak topics identified by the AI in the '/analyze' step.
    """
    # Get current subject/subtopic from session
    current_subject = session.get("current_subject")
    current_subtopic = session.get("current_subtopic")

    if not current_subject or not current_subtopic:
        app.logger.error(
            "No current subject/subtopic found in session for remedial quiz generation."
        )
        session["quiz_generation_error"] = (
            "Session error: Please take the main quiz first."
        )
        return redirect(url_for("show_results_page"))

    # Get weak topics with subject/subtopic prefix
    weak_topics = session.get(
        get_session_key(current_subject, current_subtopic, "weak_topics"), []
    )

    # DEBUG: Log weak topics being used for lesson search
    app.logger.info(f"[DEBUG] Searching for lessons with weak topics: {weak_topics}")

    if not weak_topics:
        app.logger.info(
            f"No weak topics in session for {current_subject}/{current_subtopic}; cannot generate remedial quiz."
        )
        session["quiz_generation_error"] = (
            "You've mastered all identified topics! No remedial quiz needed."
        )
        return redirect(url_for("show_results_page"))

    # Get question pool for current subject/subtopic
    question_pool = get_question_pool(current_subject, current_subtopic)

    # Get matching lessons for weak topics
    matching_lessons = get_lessons_by_tags(current_subject, weak_topics)

    # DEBUG: Log lesson search results
    app.logger.info(
        f"[DEBUG] Found {len(matching_lessons)} matching lessons for weak topics"
    )
    for lesson in matching_lessons:
        app.logger.info(
            f"[DEBUG] - Lesson: {lesson.get('title', 'No title')} (tags: {lesson.get('tags', [])})"
        )

    # Select questions from the pool that match the weak topics
    remedial_questions = []
    selected_questions_set = set()  # To avoid duplicate questions

    app.logger.info(
        f"Filtering question pool for weak topics in {current_subject}/{current_subtopic}: {weak_topics}"
    )

    for question in question_pool:
        # Check if any of the question's tags are in the user's weak topics
        question_tags = set(question.get("tags", []))
        if not question_tags.isdisjoint(weak_topics):
            # Use the question's text as a unique identifier to avoid duplicates
            if question["question"] not in selected_questions_set:
                remedial_questions.append(question)
                selected_questions_set.add(question["question"])

    if not remedial_questions:
        app.logger.warning(
            f"No questions found in question pool for topics: {weak_topics} in {current_subject}/{current_subtopic}"
        )
        session["quiz_generation_error"] = (
            "We couldn't find specific follow-up questions for your weak topics. Please review the materials and try the main quiz again."
        )
        return redirect(url_for("show_results_page"))

    # Store the selected questions and lessons with subject/subtopic prefixes
    session[
        get_session_key(
            current_subject, current_subtopic, "current_remedial_quiz_questions"
        )
    ] = remedial_questions
    session[
        get_session_key(current_subject, current_subtopic, "recommended_lessons")
    ] = matching_lessons
    session[
        get_session_key(
            current_subject, current_subtopic, "questions_served_for_analysis"
        )
    ] = remedial_questions
    session[get_session_key(current_subject, current_subtopic, "current_quiz_type")] = (
        "remedial"
    )
    session[
        get_session_key(
            current_subject, current_subtopic, "topics_for_current_remedial_quiz"
        )
    ] = weak_topics

    app.logger.info(
        f"Selected {len(remedial_questions)} questions for the remedial quiz in {current_subject}/{current_subtopic}."
    )
    app.logger.info(
        f"Found {len(matching_lessons)} matching lessons for weak topics: {weak_topics}"
    )

    return redirect(url_for("take_remedial_quiz_page"))


@app.route("/take_remedial_quiz")
def take_remedial_quiz_page():
    # Get current subject/subtopic from session
    current_subject = session.get("current_subject")
    current_subtopic = session.get("current_subtopic")

    if not current_subject or not current_subtopic:
        app.logger.info("No current subject/subtopic in session for remedial quiz.")
        session["quiz_generation_error"] = (
            "Session error: Please take the main quiz first."
        )
        return redirect(url_for("show_results_page"))

    # Get remedial questions with subject/subtopic prefix
    remedial_questions = session.get(
        get_session_key(
            current_subject, current_subtopic, "current_remedial_quiz_questions"
        ),
        [],
    )

    if not remedial_questions:
        app.logger.info(
            f"No remedial quiz in session for {current_subject}/{current_subtopic}, redirecting to results with error."
        )
        session["quiz_generation_error"] = (
            "No remedial quiz was available to take. Perhaps try again or review more."
        )
        return redirect(url_for("show_results_page"))

    quiz_title = "Remedial Quiz"
    targeted_topics = session.get(
        get_session_key(
            current_subject, current_subtopic, "topics_for_current_remedial_quiz"
        )
    )
    if targeted_topics:
        quiz_title += " (Focusing on: " + ", ".join(targeted_topics) + ")"

    return render_template(
        "quiz.html", questions=remedial_questions, quiz_title=quiz_title
    )


@app.route("/results")
def show_results_page():
    quiz_gen_error = session.pop("quiz_generation_error", None)

    # Get current subject/subtopic from session
    current_subject = session.get("current_subject", "python")
    current_subtopic = session.get("current_subtopic", "functions")

    # Load video data using the new system
    try:
        video_data = get_video_data(current_subject, current_subtopic)
        # Convert to legacy format for compatibility
        VIDEO_DATA = {}
        if video_data:
            for key, video_info in video_data.items():
                VIDEO_DATA[key] = {
                    "title": video_info.get("title", ""),
                    "url": f"https://www.youtube.com/embed/{video_info.get('videoId', '')}?enablejsapi=1",
                    "description": video_info.get("description", ""),
                }
        else:
            # Fallback to default Python topics if no video data
            VIDEO_DATA = {
                "functions": {
                    "title": "Python Functions Masterclass",
                    "url": "https://www.youtube.com/embed/kvO_nHnvPtQ?enablejsapi=1",
                    "description": "Master Python functions, parameters, return values, and scope.",
                },
                "loops": {
                    "title": "Python Loops: For and While",
                    "url": "https://www.youtube.com/watch?v=94UHCEmprCY",
                    "description": "Learn how to automate repetitive tasks using for and while loops.",
                },
            }
    except Exception as e:
        app.logger.error(f"Error loading video data for results page: {e}")
        VIDEO_DATA = {}

    # Get recommended lessons from session if they exist
    recommended_lessons = session.get(
        get_session_key(current_subject, current_subtopic, "recommended_lessons"), []
    )

    # Get weak topics from session to organize lessons by topic
    weak_topics = session.get(
        get_session_key(current_subject, current_subtopic, "weak_topics"), []
    )

    # Transform recommended lessons into the format expected by the template
    # The template expects LESSON_PLANS[topic] to contain lesson data
    lesson_plans = {}

    if recommended_lessons and weak_topics:
        app.logger.info(
            f"[DEBUG] Processing {len(recommended_lessons)} recommended lessons for weak topics: {weak_topics}"
        )

        # Group lessons by weak topic - collect ALL matching lessons for each topic
        lessons_by_topic = {}
        for topic in weak_topics:
            lessons_by_topic[topic] = []

        # Load the actual lesson content for recommended lessons
        for lesson_info in recommended_lessons:
            subject = lesson_info.get("subject")
            subtopic = lesson_info.get("subtopic")
            lesson_id = lesson_info.get("lesson_id")
            matching_tags = lesson_info.get("matching_tags", [])

            # Load the full lesson data
            try:
                full_lesson_plans = get_lesson_plans(subject, subtopic)
                if lesson_id in full_lesson_plans:
                    lesson_data = full_lesson_plans[lesson_id]

                    # Add this lesson to ALL matching weak topics
                    for tag in matching_tags:
                        if tag in weak_topics:
                            lessons_by_topic[tag].append(lesson_data)
                            app.logger.info(
                                f"[DEBUG] Added lesson '{lesson_data.get('title', 'No title')}' for topic '{tag}'"
                            )

            except Exception as e:
                app.logger.error(
                    f"Error loading lesson content for {subject}/{subtopic}/{lesson_id}: {e}"
                )

        # For the template, we'll use the best lesson for each topic
        for topic, lessons_list in lessons_by_topic.items():
            if lessons_list:
                # Sort lessons by relevance - prefer lessons with more specific matches to the topic
                def lesson_relevance(lesson_data):
                    lesson_tags = set(lesson_data.get("tags", []))
                    # Count how many times the current topic appears in the lesson's tags
                    topic_matches = sum(
                        1
                        for tag in lesson_tags
                        if topic.lower() in tag.lower() or tag.lower() in topic.lower()
                    )
                    # Prefer lessons with more topic-specific matches
                    return topic_matches

                # Sort by relevance (descending) and take the most relevant lesson
                sorted_lessons = sorted(
                    lessons_list, key=lesson_relevance, reverse=True
                )
                lesson_plans[topic] = sorted_lessons[0]

                app.logger.info(
                    f"[DEBUG] Topic '{topic}' has {len(lessons_list)} available lessons, selected most relevant: '{sorted_lessons[0].get('title', 'No title')}'"
                )
                if len(lessons_list) > 1:
                    other_titles = [
                        lesson.get("title", "No title") for lesson in sorted_lessons[1:]
                    ]
                    app.logger.info(
                        f"[DEBUG] Other available lessons for '{topic}': {other_titles}"
                    )
            else:
                app.logger.warning(
                    f"[DEBUG] No lessons found for weak topic: '{topic}'"
                )

    # If no recommended lessons found, fall back to all lessons from current subtopic
    if not lesson_plans:
        app.logger.info(
            f"[DEBUG] No recommended lessons found, falling back to all lessons from {current_subject}/{current_subtopic}"
        )
        try:
            lesson_plans = get_lesson_plans(current_subject, current_subtopic)
        except Exception:
            lesson_plans = {}

    app.logger.info(f"[DEBUG] Final lesson_plans keys: {list(lesson_plans.keys())}")

    return render_template(
        "results.html",
        quiz_generation_error=quiz_gen_error,
        VIDEO_DATA=VIDEO_DATA,
        LESSON_PLANS=lesson_plans,
        current_subject=current_subject,
        current_subtopic=current_subtopic,
        is_admin=is_admin_override_active(session),
    )


#  ADMIN PANEL ROUTES


@app.route("/admin")
def admin_dashboard():
    """Admin dashboard overview."""
    try:
        # Auto-discover subjects
        subjects = data_loader.discover_subjects()

        # Calculate stats
        total_subjects = len(subjects)
        total_subtopics = 0
        total_lessons = 0
        total_questions = 0

        for subject_id in subjects.keys():
            try:
                subject_config = data_loader.load_subject_config(subject_id)
                if subject_config and "subtopics" in subject_config:
                    subtopics = subject_config["subtopics"]
                    total_subtopics += len(subtopics)

                    for subtopic_id, subtopic_data in subtopics.items():
                        total_lessons += subtopic_data.get("lesson_count", 0)
                        total_questions += subtopic_data.get("question_count", 0)
            except Exception as e:
                app.logger.error(f"Error loading stats for subject {subject_id}: {e}")

        stats = {
            "total_subjects": total_subjects,
            "total_subtopics": total_subtopics,
            "total_lessons": total_lessons,
            "total_questions": total_questions,
        }

        return render_template("admin/dashboard.html", subjects=subjects, stats=stats)
    except Exception as e:
        app.logger.error(f"Error loading admin dashboard: {e}")
        return f"Error loading admin dashboard: {e}", 500


@app.route("/admin/subjects")
def admin_subjects():
    """Manage subjects."""
    try:
        # Auto-discover subjects
        subjects = data_loader.discover_subjects()

        return render_template("admin/subjects.html", subjects=subjects)
    except Exception as e:
        app.logger.error(f"Error loading subjects admin: {e}")
        return f"Error loading subjects: {e}", 500


@app.route("/admin/subjects/create", methods=["GET", "POST"])
def admin_create_subject():
    """Create a new subject."""
    if request.method == "POST":
        try:
            data = request.json
            subject_id = data.get("id", "").lower().replace(" ", "_")
            subject_name = data.get("name", "")
            description = data.get("description", "")
            icon = data.get("icon", "fas fa-book")
            color = data.get("color", "#007bff")

            if not subject_id or not subject_name:
                return jsonify({"error": "Subject ID and name are required"}), 400

            # Check if subject already exists by checking for directory
            subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject_id)
            if os.path.exists(subject_dir):
                return jsonify({"error": "Subject already exists"}), 400

            # Create subject directory
            os.makedirs(subject_dir, exist_ok=True)

            # Create subject_info.json
            subject_info = {
                "name": subject_name,
                "description": description,
                "icon": icon,
                "color": color,
                "status": "active",
                "created_date": "2025-01-01",
            }

            info_path = os.path.join(subject_dir, "subject_info.json")
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(subject_info, f, indent=2)

            # Create subject_config.json
            subject_config = {
                "subtopics": {},
                "allowed_keywords": [],
            }

            config_path = os.path.join(subject_dir, "subject_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(subject_config, f, indent=2)

            # Clear cache to ensure fresh data is loaded
            data_loader.clear_cache()

            return jsonify({"success": True, "message": "Subject created successfully"})

        except Exception as e:
            app.logger.error(f"Error creating subject: {e}")
            return jsonify({"error": str(e)}), 500

    return render_template("admin/create_subject.html")


@app.route("/admin/subjects/<subject>/edit")
def admin_edit_subject(subject):
    """Edit a subject."""
    try:
        # Load both subject_info.json and subject_config.json
        subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject)
        
        # Load subject_info.json
        info_path = os.path.join(subject_dir, "subject_info.json")
        subject_info = {}
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                subject_info = json.load(f)
        
        # Load subject_config.json
        subject_config = data_loader.load_subject_config(subject)
        if not subject_config:
            return f"Subject '{subject}' not found", 404

        # Merge the data for the template (backward compatibility)
        merged_config = {
            **subject_config,
            "subject_info": subject_info
        }

        return render_template(
            "admin/edit_subject.html", subject=subject, config=merged_config
        )
    except Exception as e:
        app.logger.error(f"Error loading subject editor for {subject}: {e}")
        return f"Error: {e}", 500


@app.route("/admin/subjects/<subject>/update", methods=["POST"])
def admin_update_subject(subject):
    """Update an existing subject."""
    try:
        data = request.json
        subject_name = data.get("name", "")
        description = data.get("description", "")
        icon = data.get("icon", "fas fa-book")
        color = data.get("color", "#007bff")
        allowed_keywords = data.get("allowed_keywords", [])

        if not subject_name:
            return jsonify({"error": "Subject name is required"}), 400

        # Check if subject directory exists
        subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject)
        if not os.path.exists(subject_dir):
            return jsonify({"error": "Subject not found"}), 404

        # Update subject_info.json
        info_path = os.path.join(subject_dir, "subject_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                subject_info = json.load(f)
        else:
            # Create new subject_info.json if it doesn't exist
            subject_info = {"status": "active", "created_date": "2025-01-01"}

        # Update subject info
        subject_info.update({
            "name": subject_name,
            "description": description,
            "icon": icon,
            "color": color,
        })

        # Save subject_info.json
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(subject_info, f, indent=2)

        # Update subject_config.json
        config_path = os.path.join(subject_dir, "subject_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                subject_config = json.load(f)
        else:
            return jsonify({"error": "Subject config not found"}), 404

        # Update allowed keywords
        subject_config["allowed_keywords"] = allowed_keywords

        # Save subject config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(subject_config, f, indent=2)

        # Clear cache to ensure fresh data is loaded
        data_loader.clear_cache()

        return jsonify({"success": True, "message": "Subject updated successfully"})

    except Exception as e:
        app.logger.error(f"Error updating subject: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/subjects/<subject>/<subtopic>")
def admin_edit_subtopic(subject, subtopic):
    """Edit a subtopic."""
    try:
        subject_config = data_loader.load_subject_config(subject)
        if not subject_config or subtopic not in subject_config.get("subtopics", {}):
            return f"Subtopic '{subtopic}' not found in subject '{subject}'", 404

        subtopic_data = subject_config["subtopics"][subtopic]

        # Load additional data
        quiz_data = data_loader.load_quiz_data(subject, subtopic)
        lesson_plans = data_loader.load_lesson_plans(subject, subtopic)
        videos = data_loader.load_videos(subject, subtopic)

        return render_template(
            "admin/edit_subtopic.html",
            subject=subject,
            subtopic=subtopic,
            subtopic_data=subtopic_data,
            quiz_data=quiz_data,
            lesson_plans=lesson_plans,
            videos=videos,
        )
    except Exception as e:
        app.logger.error(f"Error loading subtopic editor for {subject}/{subtopic}: {e}")
        return f"Error: {e}", 500


@app.route("/admin/subjects/<subject>/delete", methods=["DELETE"])
def admin_delete_subject(subject):
    """Delete a subject and all its associated data."""
    try:
        # Check if subject directory exists
        subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject)
        if not os.path.exists(subject_dir):
            return jsonify({"error": "Subject not found"}), 404

        # Remove subject directory and all its contents
        shutil.rmtree(subject_dir)
        app.logger.info(f"Removed subject directory: {subject_dir}")

        # Clear cache to ensure fresh data is loaded
        data_loader.clear_cache()

        return jsonify(
            {"success": True, "message": f"Subject '{subject}' deleted successfully"}
        )

    except Exception as e:
        app.logger.error(f"Error deleting subject {subject}: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ADMIN - LESSONS MANAGEMENT ROUTES
# ============================================================================


def get_all_lessons():
    """Get all lessons across all subjects and subtopics."""
    lessons_data = []

    try:
        # Auto-discover subjects
        subjects = data_loader.discover_subjects()

        for subject_id in subjects.keys():
            subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject_id)
            if not os.path.exists(subject_dir):
                continue

            # Get all subtopics for this subject
            for item in os.listdir(subject_dir):
                subtopic_dir = os.path.join(subject_dir, item)
                if os.path.isdir(subtopic_dir):
                    lesson_plans_path = os.path.join(subtopic_dir, "lesson_plans.json")
                    if os.path.exists(lesson_plans_path):
                        with open(lesson_plans_path, "r", encoding="utf-8") as f:
                            lesson_plans = json.load(f)

                        for lesson_id, lesson_data in lesson_plans.get(
                            "lessons", {}
                        ).items():
                            lessons_data.append(
                                {
                                    "id": lesson_id,
                                    "subject": subject_id,
                                    "subtopic": item,
                                    "title": lesson_data.get("title", lesson_id),
                                    "videoId": lesson_data.get("videoId", ""),
                                    "content_count": len(
                                        lesson_data.get("content", [])
                                    ),
                                    "subject_name": subjects[subject_id].get(
                                        "name", subject_id
                                    ),
                                }
                            )
    except Exception as e:
        app.logger.error(f"Error getting all lessons: {e}")

    return lessons_data


def save_lesson_to_file(subject, subtopic, lesson_id, lesson_data):
    """Save a lesson to the lesson_plans.json file."""
    lesson_plans_path = os.path.join(
        DATA_ROOT_PATH, "subjects", subject, subtopic, "lesson_plans.json"
    )

    try:
        # Load existing lesson plans
        if os.path.exists(lesson_plans_path):
            with open(lesson_plans_path, "r", encoding="utf-8") as f:
                lesson_plans = json.load(f)
        else:
            lesson_plans = {"lessons": {}}

        # Add or update the lesson
        lesson_plans["lessons"][lesson_id] = lesson_data

        # Save back to file
        with open(lesson_plans_path, "w", encoding="utf-8") as f:
            json.dump(lesson_plans, f, indent=2)

        # Clear cache for this subject/subtopic to ensure fresh data is loaded
        data_loader.clear_cache_for_subject_subtopic(subject, subtopic)

        return True
    except Exception as e:
        app.logger.error(f"Error saving lesson {lesson_id}: {e}")
        return False


def delete_lesson_from_file(subject, subtopic, lesson_id):
    """Delete a lesson from the lesson_plans.json file."""
    lesson_plans_path = os.path.join(
        DATA_ROOT_PATH, "subjects", subject, subtopic, "lesson_plans.json"
    )

    try:
        if not os.path.exists(lesson_plans_path):
            return False

        with open(lesson_plans_path, "r", encoding="utf-8") as f:
            lesson_plans = json.load(f)

        if lesson_id in lesson_plans.get("lessons", {}):
            del lesson_plans["lessons"][lesson_id]

            with open(lesson_plans_path, "w", encoding="utf-8") as f:
                json.dump(lesson_plans, f, indent=2)

            # Clear cache for this subject/subtopic to ensure fresh data is loaded
            data_loader.clear_cache_for_subject_subtopic(subject, subtopic)
            return True
        return False
    except Exception as e:
        app.logger.error(f"Error deleting lesson {lesson_id}: {e}")
        return False


# ==================== SUBTOPICS ADMIN ROUTES ====================


@app.route("/admin/subtopics")
def admin_subtopics():
    """Manage subtopics across all subjects."""
    try:
        # Auto-discover subjects
        subjects = data_loader.discover_subjects()

        # Enhance subjects data with subtopic information from subject_config.json
        for subject_id, subject_info in subjects.items():
            subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject_id)
            subject_config_path = os.path.join(subject_dir, "subject_config.json")

            if os.path.exists(subject_config_path):
                try:
                    with open(subject_config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)

                    # Get subtopics from subject_config.json
                    config_subtopics = config.get("subtopics", {})
                    allowed_keywords = config.get("allowed_keywords", [])

                    # Also check actual directory structure for any additional subtopics
                    actual_subtopics = {}
                    if os.path.exists(subject_dir):
                        for item in os.listdir(subject_dir):
                            subtopic_dir = os.path.join(subject_dir, item)
                            if os.path.isdir(subtopic_dir) and item != "__pycache__":
                                # Count lessons and questions
                                lesson_plans_path = os.path.join(
                                    subtopic_dir, "lesson_plans.json"
                                )
                                question_pool_path = os.path.join(
                                    subtopic_dir, "question_pool.json"
                                )

                                lesson_count = 0
                                question_count = 0

                                if os.path.exists(lesson_plans_path):
                                    try:
                                        with open(
                                            lesson_plans_path, "r", encoding="utf-8"
                                        ) as f:
                                            lessons = json.load(f)
                                            lesson_count = len(
                                                lessons.get("lessons", {})
                                            )
                                    except:
                                        pass

                                if os.path.exists(question_pool_path):
                                    try:
                                        with open(
                                            question_pool_path, "r", encoding="utf-8"
                                        ) as f:
                                            questions = json.load(f)
                                            question_count = len(
                                                questions.get("questions", {})
                                            )
                                    except:
                                        pass

                                actual_subtopics[item] = {
                                    "lesson_count": lesson_count,
                                    "question_count": question_count,
                                }

                    # Merge config subtopics with actual directory info
                    merged_subtopics = {}
                    for subtopic_id, subtopic_info in config_subtopics.items():
                        merged_info = dict(subtopic_info)  # Copy config info
                        if subtopic_id in actual_subtopics:
                            merged_info.update(actual_subtopics[subtopic_id])
                        else:
                            merged_info.update({"lesson_count": 0, "question_count": 0})
                        merged_subtopics[subtopic_id] = merged_info

                    # Add any directory-only subtopics not in config
                    for subtopic_id, counts in actual_subtopics.items():
                        if subtopic_id not in merged_subtopics:
                            merged_subtopics[subtopic_id] = {
                                "name": subtopic_id.replace("-", " ").title(),
                                "description": "",
                                "order": 999,
                                "status": "active",
                                "prerequisites": [],
                                "estimated_time": "",
                                "video_count": 0,
                                **counts,
                            }

                    subjects[subject_id]["subtopics"] = merged_subtopics
                    subjects[subject_id]["allowed_keywords"] = allowed_keywords

                except Exception as e:
                    app.logger.error(
                        f"Error reading subject config for {subject_id}: {e}"
                    )
                    subjects[subject_id]["subtopics"] = {}
                    subjects[subject_id]["allowed_keywords"] = []
            else:
                subjects[subject_id]["subtopics"] = {}
                subjects[subject_id]["allowed_keywords"] = []

        return render_template("admin/subtopics.html", subjects=subjects)

    except Exception as e:
        app.logger.error(f"Error loading subtopics: {e}")
        return render_template("admin/subtopics.html", subjects={})


@app.route("/admin/subtopics", methods=["POST"])
def admin_create_subtopic():
    """Create a new subtopic."""
    try:
        data = request.get_json()
        subject = data.get("subject")
        subtopic_id = data.get("subtopic_id")
        name = data.get("name", "")
        description = data.get("description", "")
        keywords = data.get("keywords", [])
        estimated_time = data.get("estimated_time", "")
        order = data.get("order", 1)
        prerequisites = data.get("prerequisites", [])
        video_data = data.get("video")  # New video data

        # Validation
        if not subject or not subtopic_id:
            return jsonify({"error": "Subject and subtopic ID are required"}), 400

        # Validate subtopic ID format
        if not re.match(r"^[a-z0-9-]+$", subtopic_id):
            return (
                jsonify(
                    {
                        "error": "Subtopic ID can only contain lowercase letters, numbers, and hyphens"
                    }
                ),
                400,
            )

        # Check if subject exists by checking for subject directory and files
        subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject)
        subject_config_path = os.path.join(subject_dir, "subject_config.json")
        subject_info_path = os.path.join(subject_dir, "subject_info.json")
        
        if not (os.path.exists(subject_config_path) and os.path.exists(subject_info_path)):
            return jsonify({"error": "Subject not found"}), 404

        # Load subject config
        with open(subject_config_path, "r", encoding="utf-8") as f:
            subject_config = json.load(f)

        # Check if subtopic already exists
        if subtopic_id in subject_config.get("subtopics", {}):
            return jsonify({"error": "Subtopic already exists"}), 400

        # Validate prerequisites exist in the same subject
        if prerequisites:
            existing_subtopics = set(subject_config.get("subtopics", {}).keys())
            invalid_prerequisites = [
                prereq for prereq in prerequisites if prereq not in existing_subtopics
            ]
            if invalid_prerequisites:
                return (
                    jsonify(
                        {
                            "error": f"Invalid prerequisites: {', '.join(invalid_prerequisites)}"
                        }
                    ),
                    400,
                )

        # Create subtopic directory
        subtopic_dir = os.path.join(subject_dir, subtopic_id)
        os.makedirs(subtopic_dir, exist_ok=True)

        # Add subtopic to subject config
        if "subtopics" not in subject_config:
            subject_config["subtopics"] = {}

        subject_config["subtopics"][subtopic_id] = {
            "name": name or subtopic_id.replace("-", " ").title(),
            "description": description,
            "order": order,
            "status": "active",
            "prerequisites": prerequisites,
            "estimated_time": estimated_time,
            "video_count": 0,
            "lesson_count": 0,
            "question_count": 0,
        }

        # Update allowed keywords if provided
        if keywords:
            if "allowed_keywords" not in subject_config:
                subject_config["allowed_keywords"] = []

            # Add new keywords that don't already exist
            existing_keywords = set(subject_config["allowed_keywords"])
            for keyword in keywords:
                if keyword.strip() and keyword.strip().lower() not in [
                    k.lower() for k in existing_keywords
                ]:
                    subject_config["allowed_keywords"].append(keyword.strip())

        # Save updated subject config
        with open(subject_config_path, "w", encoding="utf-8") as f:
            json.dump(subject_config, f, indent=2)

        # Create empty lesson plans file
        lesson_plans_data = {"lessons": {}}
        lesson_plans_path = os.path.join(subtopic_dir, "lesson_plans.json")
        with open(lesson_plans_path, "w", encoding="utf-8") as f:
            json.dump(lesson_plans_data, f, indent=2)

        # Create empty question pool file
        question_pool_data = {"questions": []}
        question_pool_path = os.path.join(subtopic_dir, "question_pool.json")
        with open(question_pool_path, "w", encoding="utf-8") as f:
            json.dump(question_pool_data, f, indent=2)

        # Create empty quiz data file
        subtopic_title = name or subtopic_id.replace("-", " ").title()
        quiz_data = {
            "quiz_title": f"{subject.title()} {subtopic_title} Quiz",
            "questions": [],
        }
        quiz_path = os.path.join(subtopic_dir, "quiz_data.json")
        with open(quiz_path, "w", encoding="utf-8") as f:
            json.dump(quiz_data, f, indent=2)

        # Create videos file with video data if provided
        videos_data = {"videos": {}}
        if video_data:
            videos_data["videos"][subtopic_id] = {
                "title": video_data.get("title", ""),
                "url": video_data.get("url", ""),
                "description": video_data.get("description", ""),
            }
            # Update video count in subtopic config
            subject_config["subtopics"][subtopic_id]["video_count"] = 1

        videos_path = os.path.join(subtopic_dir, "videos.json")
        with open(videos_path, "w", encoding="utf-8") as f:
            json.dump(videos_data, f, indent=2)

        # Clear cache to ensure fresh data is loaded
        data_loader.clear_cache()

        return jsonify(
            {
                "success": True,
                "message": f"Subtopic '{subject_config['subtopics'][subtopic_id]['name']}' created successfully",
            }
        )

    except Exception as e:
        app.logger.error(f"Error creating subtopic: {e}")
        return jsonify({"error": "An error occurred while creating the subtopic"}), 500


@app.route("/admin/subtopics/<subject>/<subtopic_id>", methods=["PUT"])
def admin_update_subtopic(subject, subtopic_id):
    """Update an existing subtopic."""
    try:
        data = request.get_json()
        name = data.get("name", "")
        description = data.get("description", "")
        keywords = data.get("keywords", [])
        estimated_time = data.get("estimated_time", "")
        order = data.get("order", 1)
        prerequisites = data.get("prerequisites", [])
        video_data = data.get("video")  # New video data

        # Load subject config
        subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject)
        subject_config_path = os.path.join(subject_dir, "subject_config.json")

        if not os.path.exists(subject_config_path):
            return jsonify({"error": "Subject configuration not found"}), 404

        with open(subject_config_path, "r", encoding="utf-8") as f:
            subject_config = json.load(f)

        # Check if subtopic exists
        if subtopic_id not in subject_config.get("subtopics", {}):
            return jsonify({"error": "Subtopic not found"}), 404

        # Validate prerequisites exist in the same subject and don't create circular dependencies
        if prerequisites:
            existing_subtopics = set(subject_config.get("subtopics", {}).keys())
            invalid_prerequisites = [
                prereq for prereq in prerequisites if prereq not in existing_subtopics
            ]
            if invalid_prerequisites:
                return (
                    jsonify(
                        {
                            "error": f"Invalid prerequisites: {', '.join(invalid_prerequisites)}"
                        }
                    ),
                    400,
                )

            # Check for circular dependencies
            if subtopic_id in prerequisites:
                return (
                    jsonify({"error": "A subtopic cannot be a prerequisite of itself"}),
                    400,
                )

        # Update subtopic information
        subject_config["subtopics"][subtopic_id].update(
            {
                "name": name or subtopic_id.replace("-", " ").title(),
                "description": description,
                "estimated_time": estimated_time,
                "order": order,
                "prerequisites": prerequisites,
            }
        )

        # Update allowed keywords if provided
        if keywords:
            if "allowed_keywords" not in subject_config:
                subject_config["allowed_keywords"] = []

            # Add new keywords that don't already exist
            existing_keywords = set(subject_config["allowed_keywords"])
            for keyword in keywords:
                if keyword.strip() and keyword.strip().lower() not in [
                    k.lower() for k in existing_keywords
                ]:
                    subject_config["allowed_keywords"].append(keyword.strip())

        # Handle video data update
        if video_data is not None:  # Check for None to allow clearing video data
            subtopic_dir = os.path.join(
                DATA_ROOT_PATH, "subjects", subject, subtopic_id
            )
            videos_path = os.path.join(subtopic_dir, "videos.json")

            # Load existing videos or create new structure
            videos_data = {"videos": {}}
            if os.path.exists(videos_path):
                try:
                    with open(videos_path, "r", encoding="utf-8") as f:
                        videos_data = json.load(f)
                except:
                    videos_data = {"videos": {}}

            # Update video data
            if video_data:  # If video data provided
                videos_data["videos"][subtopic_id] = {
                    "title": video_data.get("title", ""),
                    "url": video_data.get("url", ""),
                    "description": video_data.get("description", ""),
                }
                subject_config["subtopics"][subtopic_id]["video_count"] = 1
            else:  # If video data is empty (clearing video)
                if subtopic_id in videos_data.get("videos", {}):
                    del videos_data["videos"][subtopic_id]
                subject_config["subtopics"][subtopic_id]["video_count"] = 0

            # Save videos file
            with open(videos_path, "w", encoding="utf-8") as f:
                json.dump(videos_data, f, indent=2)

        # Save updated subject config
        with open(subject_config_path, "w", encoding="utf-8") as f:
            json.dump(subject_config, f, indent=2)

        # Clear cache to ensure fresh data is loaded
        data_loader.clear_cache()

        return jsonify(
            {
                "success": True,
                "message": f"Subtopic '{subject_config['subtopics'][subtopic_id]['name']}' updated successfully",
            }
        )

    except Exception as e:
        app.logger.error(f"Error updating subtopic: {e}")
        return jsonify({"error": "An error occurred while updating the subtopic"}), 500


@app.route("/admin/subtopics/<subject>/<subtopic_id>", methods=["DELETE"])
def admin_delete_subtopic(subject, subtopic_id):
    """Delete a subtopic and all its associated data."""
    try:
        # Load subject config
        subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject)
        subject_config_path = os.path.join(subject_dir, "subject_config.json")

        if not os.path.exists(subject_config_path):
            return jsonify({"error": "Subject configuration not found"}), 404

        with open(subject_config_path, "r", encoding="utf-8") as f:
            subject_config = json.load(f)

        # Check if subtopic exists in config
        subtopic_name = subtopic_id.replace("-", " ").title()
        if subtopic_id in subject_config.get("subtopics", {}):
            subtopic_name = subject_config["subtopics"][subtopic_id].get(
                "name", subtopic_name
            )
            # Remove from config
            del subject_config["subtopics"][subtopic_id]

            # Save updated config
            with open(subject_config_path, "w", encoding="utf-8") as f:
                json.dump(subject_config, f, indent=2)

        # Remove the subtopic directory if it exists
        subtopic_dir = os.path.join(subject_dir, subtopic_id)
        if os.path.exists(subtopic_dir):
            shutil.rmtree(subtopic_dir)

        # Clear cache for this subject/subtopic
        data_loader.clear_cache_for_subject_subtopic(subject, subtopic_id)

        return jsonify(
            {
                "success": True,
                "message": f"Subtopic '{subtopic_name}' and all associated data deleted successfully",
            }
        )

    except Exception as e:
        app.logger.error(f"Error deleting subtopic: {e}")
        return jsonify({"error": "An error occurred while deleting the subtopic"}), 500


@app.route("/admin/subjects/<subject>/keywords", methods=["GET"])
def admin_get_subject_keywords(subject):
    """Get keywords for a specific subject."""
    try:
        subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject)
        subject_config_path = os.path.join(subject_dir, "subject_config.json")

        if not os.path.exists(subject_config_path):
            return jsonify({"error": "Subject configuration not found"}), 404

        with open(subject_config_path, "r", encoding="utf-8") as f:
            subject_config = json.load(f)

        keywords = subject_config.get("allowed_keywords", [])
        return jsonify({"keywords": keywords})

    except Exception as e:
        app.logger.error(f"Error getting subject keywords: {e}")
        return jsonify({"error": "An error occurred while retrieving keywords"}), 500


@app.route("/admin/subjects/<subject>/keywords", methods=["PUT"])
def admin_update_subject_keywords(subject):
    """Update keywords for a subject."""
    try:
        data = request.get_json()
        keywords = data.get("keywords", [])

        # Load subject config
        subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject)
        subject_config_path = os.path.join(subject_dir, "subject_config.json")

        if not os.path.exists(subject_config_path):
            return jsonify({"error": "Subject configuration not found"}), 404

        with open(subject_config_path, "r", encoding="utf-8") as f:
            subject_config = json.load(f)

        # Update allowed keywords
        subject_config["allowed_keywords"] = [k.strip() for k in keywords if k.strip()]

        # Save updated subject config
        with open(subject_config_path, "w", encoding="utf-8") as f:
            json.dump(subject_config, f, indent=2)

        return jsonify(
            {
                "success": True,
                "message": f"Keywords updated successfully. {len(subject_config['allowed_keywords'])} keywords saved.",
            }
        )

    except Exception as e:
        app.logger.error(f"Error updating subject keywords: {e}")
        return jsonify({"error": "An error occurred while updating keywords"}), 500


@app.route("/admin/lessons")
def admin_lessons():
    """List all lessons across all subjects."""
    lessons = get_all_lessons()

    # Get subjects for dropdown using auto-discovery
    subjects = data_loader.discover_subjects()

    return render_template("admin/lessons.html", lessons=lessons, subjects=subjects)


@app.route("/admin/lessons/create", methods=["GET", "POST"])
def admin_create_lesson():
    """Create a new lesson."""
    if request.method == "POST":
        try:
            data = request.json
            subject = data.get("subject", "")
            subtopic = data.get("subtopic", "")
            lesson_id = data.get("lesson_id", "").lower().replace(" ", "_")
            title = data.get("title", "")
            video_id = data.get("videoId", "")
            content = data.get("content", [])
            tags = data.get("tags", [])

            if not all([subject, subtopic, lesson_id, title]):
                return (
                    jsonify(
                        {
                            "error": "Subject, subtopic, lesson ID, and title are required"
                        }
                    ),
                    400,
                )

            # Validate tags against allowed keywords
            if not tags:
                return jsonify({"error": "At least one tag is required"}), 400

            allowed_keywords = get_subject_keywords(subject)
            invalid_tags = [tag for tag in tags if tag not in allowed_keywords]
            if invalid_tags:
                return (
                    jsonify(
                        {
                            "error": f"Invalid tags: {', '.join(invalid_tags)}. Must use allowed keywords for this subject."
                        }
                    ),
                    400,
                )

            # Check if lesson already exists
            lesson_plans_path = os.path.join(
                DATA_ROOT_PATH, "subjects", subject, subtopic, "lesson_plans.json"
            )
            if os.path.exists(lesson_plans_path):
                with open(lesson_plans_path, "r", encoding="utf-8") as f:
                    existing_lessons = json.load(f).get("lessons", {})
                if lesson_id in existing_lessons:
                    return jsonify({"error": "Lesson ID already exists"}), 400

            # Create lesson data
            lesson_data = {
                "title": title,
                "videoId": video_id,
                "content": content,
                "tags": tags,
            }

            # Save lesson
            if save_lesson_to_file(subject, subtopic, lesson_id, lesson_data):
                return jsonify(
                    {"success": True, "message": "Lesson created successfully"}
                )
            else:
                return jsonify({"error": "Failed to save lesson"}), 500

        except Exception as e:
            app.logger.error(f"Error creating lesson: {e}")
            return jsonify({"error": str(e)}), 500

    # GET request - show create form using auto-discovery
    subjects = data_loader.discover_subjects()

    return render_template(
        "admin/create_lesson.html", subjects=subjects, edit_mode=False
    )


@app.route(
    "/admin/lessons/<subject>/<subtopic>/<lesson_id>/edit", methods=["GET", "POST"]
)
def admin_edit_lesson(subject, subtopic, lesson_id):
    """Edit an existing lesson."""
    if request.method == "POST":
        try:
            data = request.json
            title = data.get("title", "")
            video_id = data.get("videoId", "")
            content = data.get("content", [])
            tags = data.get("tags", [])

            if not title:
                return jsonify({"error": "Title is required"}), 400

            # Validate tags against allowed keywords
            if not tags:
                return jsonify({"error": "At least one tag is required"}), 400

            allowed_keywords = get_subject_keywords(subject)
            invalid_tags = [tag for tag in tags if tag not in allowed_keywords]
            if invalid_tags:
                return (
                    jsonify(
                        {
                            "error": f"Invalid tags: {', '.join(invalid_tags)}. Must use allowed keywords for this subject."
                        }
                    ),
                    400,
                )

            # Create lesson data
            lesson_data = {
                "title": title,
                "videoId": video_id,
                "content": content,
                "tags": tags,
            }

            # Save lesson
            if save_lesson_to_file(subject, subtopic, lesson_id, lesson_data):
                return jsonify(
                    {"success": True, "message": "Lesson updated successfully"}
                )
            else:
                return jsonify({"error": "Failed to update lesson"}), 500

        except Exception as e:
            app.logger.error(f"Error updating lesson: {e}")
            return jsonify({"error": str(e)}), 500

    # GET request - show edit form
    try:
        lesson_plans_path = os.path.join(
            DATA_ROOT_PATH, "subjects", subject, subtopic, "lesson_plans.json"
        )
        if not os.path.exists(lesson_plans_path):
            return "Lesson not found", 404

        with open(lesson_plans_path, "r", encoding="utf-8") as f:
            lesson_plans = json.load(f)

        lesson_data = lesson_plans.get("lessons", {}).get(lesson_id)
        if not lesson_data:
            return "Lesson not found", 404

        # Get subjects for context using auto-discovery
        subjects = data_loader.discover_subjects()

        # Get subtopics for the current subject
        subject_subtopics = {}
        subject_config = data_loader.load_subject_config(subject)
        if subject_config and "subtopics" in subject_config:
            for subtopic_id, subtopic_data in subject_config["subtopics"].items():
                subject_subtopics[subtopic_id] = {
                    "name": subtopic_data.get(
                        "name", subtopic_id.replace("-", " ").title()
                    ),
                    "order": subtopic_data.get("order", 0),
                }

        return render_template(
            "admin/create_lesson.html",
            subjects=subjects,
            edit_mode=True,
            lesson_data=lesson_data,
            subject=subject,
            subtopic=subtopic,
            lesson_id=lesson_id,
            subject_subtopics=subject_subtopics,
        )

    except Exception as e:
        app.logger.error(f"Error loading lesson for edit: {e}")
        return "Error loading lesson", 500


@app.route("/admin/lessons/<subject>/<subtopic>/<lesson_id>/delete", methods=["DELETE"])
def admin_delete_lesson(subject, subtopic, lesson_id):
    """Delete a lesson."""
    try:
        if delete_lesson_from_file(subject, subtopic, lesson_id):
            return jsonify({"success": True, "message": "Lesson deleted successfully"})
        else:
            return jsonify({"error": "Lesson not found or could not be deleted"}), 404

    except Exception as e:
        app.logger.error(f"Error deleting lesson: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/subjects/<subject_id>/subtopics", methods=["GET"])
def get_subject_subtopics(subject_id):
    """Get subtopics for a specific subject."""
    try:
        subject_config = data_loader.load_subject_config(subject_id)
        if not subject_config or "subtopics" not in subject_config:
            return jsonify({"subtopics": {}})

        # Return only the subtopic names and IDs for the dropdown
        subtopics = {}
        for subtopic_id, subtopic_data in subject_config["subtopics"].items():
            subtopics[subtopic_id] = {
                "name": subtopic_data.get(
                    "name", subtopic_id.replace("-", " ").title()
                ),
                "order": subtopic_data.get("order", 0),
            }

        return jsonify({"subtopics": subtopics})
    except Exception as e:
        app.logger.error(f"Error loading subtopics for subject {subject_id}: {e}")
        return jsonify({"error": "Failed to load subtopics"}), 500


@app.route("/admin/lessons/<subject>/<subtopic>")
def admin_lessons_by_subtopic(subject, subtopic):
    """View lessons for a specific subtopic."""
    try:
        lesson_plans_path = os.path.join(
            DATA_ROOT_PATH, "subjects", subject, subtopic, "lesson_plans.json"
        )
        if not os.path.exists(lesson_plans_path):
            lessons = {}
        else:
            with open(lesson_plans_path, "r", encoding="utf-8") as f:
                lessons = json.load(f).get("lessons", {})

        # Get subject info
        subjects_path = os.path.join(DATA_ROOT_PATH, "subjects.json")
        subject_info = {}
        if os.path.exists(subjects_path):
            with open(subjects_path, "r", encoding="utf-8") as f:
                subjects = json.load(f).get("subjects", {})
                subject_info = subjects.get(subject, {})

        return render_template(
            "admin/lessons.html",
            lessons=lessons,
            subject=subject,
            subtopic=subtopic,
            subject_info=subject_info,
            filtered_view=True,
        )

    except Exception as e:
        app.logger.error(f"Error loading lessons for {subject}/{subtopic}: {e}")
        return "Error loading lessons", 500


# ===== QUIZ MANAGEMENT ROUTES =====


@app.route("/admin/questions")
def admin_questions():
    """Questions management page."""
    try:
        # Auto-discover subjects and load their subtopics with quiz data
        subjects_data = {}
        stats = {
            "total_initial_questions": 0,
            "total_pool_questions": 0,
            "total_subtopics": 0,
            "subtopics_without_questions": 0,
        }

        # Get all subjects using auto-discovery
        all_subjects = data_loader.discover_subjects()

        for subject_id, subject_info in all_subjects.items():
            # Load subject config to get subtopics
            subject_config = data_loader.load_subject_config(subject_id)
            if subject_config and "subtopics" in subject_config:
                subject_data = {
                    **subject_info,
                    "subtopics": {}
                }

                for subtopic_id, subtopic_data in subject_config["subtopics"].items():
                    # Load quiz data and question pool to get counts
                    quiz_data = data_loader.load_quiz_data(subject_id, subtopic_id)
                    pool_data = data_loader.load_question_pool(subject_id, subtopic_id)

                    quiz_count = (
                        len(quiz_data.get("questions", [])) if quiz_data else 0
                    )
                    pool_count = (
                        len(pool_data.get("questions", [])) if pool_data else 0
                    )

                    subtopic_data["quiz_questions_count"] = quiz_count
                    subtopic_data["pool_questions_count"] = pool_count

                    # Update statistics
                    stats["total_initial_questions"] += quiz_count
                    stats["total_pool_questions"] += pool_count
                    stats["total_subtopics"] += 1

                    if quiz_count == 0 and pool_count == 0:
                        stats["subtopics_without_questions"] += 1

                    subject_data["subtopics"][subtopic_id] = subtopic_data

                subjects_data[subject_id] = subject_data

        return render_template(
            "admin/questions.html", subjects=subjects_data, stats=stats
        )

    except Exception as e:
        app.logger.error(f"Error loading questions admin page: {e}")
        return f"Error: {e}", 500


@app.route("/admin/quiz/<subject>/<subtopic>")
def admin_quiz_editor(subject, subtopic):
    """Edit quiz for a specific subject/subtopic."""
    try:
        # Load subject config to verify subtopic exists
        subject_config = data_loader.load_subject_config(subject)
        if not subject_config or subtopic not in subject_config.get("subtopics", {}):
            return f"Subtopic '{subtopic}' not found in subject '{subject}'", 404

        # Load quiz data (initial quiz)
        quiz_data = data_loader.load_quiz_data(subject, subtopic)
        if not quiz_data:
            quiz_data = {
                "quiz_title": f"{subject.title()} {subtopic.replace('-', ' ').title()} Quiz",
                "questions": [],
            }

        # Load question pool (remedial quiz questions)
        question_pool = data_loader.load_question_pool(subject, subtopic)
        if not question_pool:
            question_pool = {"questions": []}

        # Get subtopic info
        subtopic_data = subject_config["subtopics"][subtopic]

        return render_template(
            "admin/quiz_editor.html",
            subject=subject,
            subtopic=subtopic,
            subtopic_data=subtopic_data,
            quiz_data=quiz_data,
            question_pool=question_pool,
        )
    except Exception as e:
        app.logger.error(f"Error loading quiz editor for {subject}/{subtopic}: {e}")
        return f"Error: {e}", 500


@app.route("/admin/quiz/<subject>/<subtopic>/initial", methods=["POST"])
def admin_save_initial_quiz(subject, subtopic):
    """Save initial quiz data (quiz_data.json)."""
    try:
        data = request.json
        quiz_title = data.get("quiz_title", "")
        questions = data.get("questions", [])

        # Validate data
        if not quiz_title:
            return jsonify({"error": "Quiz title is required"}), 400

        # Ensure subtopic directory exists
        subtopic_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject, subtopic)
        os.makedirs(subtopic_dir, exist_ok=True)

        # Save quiz data
        quiz_data = {"quiz_title": quiz_title, "questions": questions}

        quiz_path = os.path.join(subtopic_dir, "quiz_data.json")
        with open(quiz_path, "w", encoding="utf-8") as f:
            json.dump(quiz_data, f, indent=2, ensure_ascii=False)

        # Clear cache
        data_loader.clear_cache()

        return jsonify({"success": True, "message": "Initial quiz saved successfully"})

    except Exception as e:
        app.logger.error(f"Error saving initial quiz for {subject}/{subtopic}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/quiz/<subject>/<subtopic>/pool", methods=["POST"])
def admin_save_question_pool(subject, subtopic):
    """Save question pool data (question_pool.json)."""
    try:
        data = request.json
        questions = data.get("questions", [])

        # Ensure subtopic directory exists
        subtopic_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject, subtopic)
        os.makedirs(subtopic_dir, exist_ok=True)

        # Save question pool
        question_pool_data = {"questions": questions}

        pool_path = os.path.join(subtopic_dir, "question_pool.json")
        with open(pool_path, "w", encoding="utf-8") as f:
            json.dump(question_pool_data, f, indent=2, ensure_ascii=False)

        # Clear cache
        data_loader.clear_cache()

        return jsonify({"success": True, "message": "Question pool saved successfully"})

    except Exception as e:
        app.logger.error(f"Error saving question pool for {subject}/{subtopic}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/quiz/validate", methods=["POST"])
def admin_validate_question():
    """Validate a quiz question structure."""
    try:
        question = request.json

        # Basic validation
        if not question.get("question"):
            return jsonify({"valid": False, "error": "Question text is required"})

        question_type = question.get("type", "multiple_choice")

        if question_type == "multiple_choice":
            options = question.get("options", [])
            answer_index = question.get("answer_index")

            if len(options) < 2:
                return jsonify(
                    {
                        "valid": False,
                        "error": "Multiple choice questions need at least 2 options",
                    }
                )

            if answer_index is None or answer_index < 0 or answer_index >= len(options):
                return jsonify(
                    {
                        "valid": False,
                        "error": "Invalid answer index for multiple choice question",
                    }
                )

        elif question_type == "fill_in_the_blank":
            if "____" not in question.get("question", ""):
                return jsonify(
                    {
                        "valid": False,
                        "error": "Fill in the blank questions must contain '____' placeholder",
                    }
                )

            if not question.get("correct_answer"):
                return jsonify(
                    {
                        "valid": False,
                        "error": "Fill in the blank questions must have a correct_answer",
                    }
                )

        elif question_type == "coding":
            if not question.get("sample_solution"):
                return jsonify(
                    {
                        "valid": False,
                        "error": "Coding questions should have a sample_solution",
                    }
                )

        # Validate tags
        tags = question.get("tags", [])
        if not tags:
            return jsonify(
                {"valid": False, "error": "Questions should have at least one tag"}
            )

        return jsonify({"valid": True, "message": "Question is valid"})

    except Exception as e:
        app.logger.error(f"Error validating question: {e}")
        return jsonify({"valid": False, "error": str(e)})


@app.route("/admin/clear-cache", methods=["POST"])
def admin_clear_cache():
    """Clear all cached data."""
    try:
        data_loader.clear_cache()
        return jsonify({"success": True, "message": "Cache cleared successfully"})
    except Exception as e:
        app.logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/toggle-override", methods=["GET", "POST"])
def admin_toggle_override():
    """Toggle admin override for testing prerequisites."""
    try:
        if request.method == "GET":
            # Return current override status
            current_state = session.get("admin_override", False)
            return jsonify({"success": True, "admin_override": current_state})

        # POST - Toggle the override
        current_state = session.get("admin_override", False)
        new_state = not current_state
        session["admin_override"] = new_state

        status = "enabled" if new_state else "disabled"
        return jsonify(
            {
                "success": True,
                "message": f"Admin override {status}",
                "admin_override": new_state,
            }
        )
    except Exception as e:
        app.logger.error(f"Error toggling admin override: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/videos/<subject>/<subtopic>", methods=["GET"])
def admin_get_video_data(subject, subtopic):
    """Get video data for a specific subject/subtopic for editing."""
    try:
        # Load video data
        video_data = get_video_data(subject, subtopic)

        # Check if video exists for this subtopic
        if subtopic in video_data:
            return jsonify({"success": True, "video": video_data[subtopic]})
        else:
            return jsonify({"success": True, "video": None})

    except Exception as e:
        app.logger.error(f"Error loading video data for {subject}/{subtopic}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    if not os.getenv("OPEN_API_KEY"):
        print(
            "ERROR: OPEN_API_KEY environment variable not set. AI features will not work."
        )

    # Validate that we have the required data structure
    if not data_loader.validate_subject_subtopic("python", "functions"):
        print(
            "ERROR: Python functions data not found. Check data/subjects/python/functions/ directory."
        )

    app.run(debug=True)
