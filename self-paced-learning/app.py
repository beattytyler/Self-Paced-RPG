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
    flash,
)  # Added redirect, url_for
from openai import OpenAI
from dotenv import load_dotenv
from utils.data_loader import DataLoader
from werkzeug.security import generate_password_hash, check_password_hash
import random, string
from flask_sqlalchemy import SQLAlchemy

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
    
app.config['TEMPLATES_AUTO_RELOAD'] = True
SECRET_KEY = os.getenv("SECRET_KEY", "devkey")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Ensure OPENAI_API_KEY is set in your .env file
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
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


def get_subject_tags(subject: str) -> list:
    """Get allowed AI analysis tags for a subject."""
    return data_loader.get_subject_keywords(subject)


def get_quiz_data(subject: str, subtopic: str) -> list:
    """Get quiz questions for a subject/subtopic."""
    return data_loader.get_quiz_questions(subject, subtopic)


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
    prompt_text, system_message, model="gpt-4", max_tokens=800, expect_json_output=False
):
    """Helper function to call the OpenAI API."""
    if not client:
        app.logger.error("OpenAI client not initialized. Cannot make API call.")
        return None  # Or raise an exception
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_text},
        ]

        completion_args = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # For newer models that support JSON mode directly
        # Check OpenAI documentation for the latest models supporting this.
        # E.g., "gpt-4-1106-preview", "gpt-3.5-turbo-1106", "gpt-4-turbo-preview", "gpt-4o"
        if expect_json_output and (
            "1106" in model
            or "turbo-preview" in model
            or "gpt-4o" in model
            or "gpt-4-turbo" in model
        ):
            completion_args["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**completion_args)
        return response.choices[0].message.content.strip()
    except Exception as e:
        app.logger.error(f"OpenAI API call failed: {e}")
        return None
    
# -------------------------
# Models
# -------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'teacher' or 'student'
    classes = db.relationship("Class", backref="teacher", lazy=True)

class Class(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(10), unique=True, nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

class ClassRegistration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey("class.id"), nullable=False)

# -------------------------
# Helper Functions
# -------------------------
def generate_class_code(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# -------------------------
@app.route('/')
def index():
    if session.get('user_id'):
        return redirect(url_for('subject_selection'))
    return redirect('/login')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        role = request.form['role']

        if User.query.filter_by(email=email).first():
            flash("Email already registered")
            return redirect('/register')

        user = User(username=username, email=email, password_hash=password, role=role)
        db.session.add(user)
        db.session.commit()
        flash('Account created! Please log in.')
        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['role'] = user.role
            flash(f"Welcome {user.username}!")

            return redirect(url_for('subject_selection'))

        flash('Invalid email or password', 'error')
    version = '1.0.3'
    return render_template('login.html', version=version)

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully")
    return redirect('/login')

#  Main Application Routes
@app.route("/subjects")
def subject_selection():
    """New home page showing all available subjects."""
    
    try:
        # Use auto-discovery instead of subjects.json
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
        # Load subject configuration and info
        subject_config = data_loader.load_subject_config(subject)
        subject_info = data_loader.load_subject_info(subject)

        if not subject_config or not subject_info:
            app.logger.error(f"Subject data not found for: {subject}")
            return redirect(url_for("subject_selection"))

        subtopics = subject_config.get("subtopics", {})

        # Calculate actual counts for each subtopic by checking the files
        for subtopic_id, subtopic_data in subtopics.items():
            try:
                # Count quiz questions
                quiz_data = get_quiz_data(subject, subtopic_id)
                question_count = len(quiz_data) if quiz_data else 0

                # Count lesson plans
                lesson_plans = get_lesson_plans(subject, subtopic_id)
                lesson_count = len(lesson_plans) if lesson_plans else 0

                # Count videos
                video_data = get_video_data(subject, subtopic_id)
                video_count = len(video_data) if video_data else 0

                # Update the counts with actual values
                subtopic_data["question_count"] = question_count
                subtopic_data["lesson_count"] = lesson_count
                subtopic_data["video_count"] = video_count

            except Exception as e:
                app.logger.warning(
                    f"Error calculating counts for {subject}/{subtopic_id}: {e}"
                )
                # Keep original counts if there's an error
                pass

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


@app.route("/quiz/<subject>/<subtopic>")
def quiz_page(subject, subtopic):
    """Serves the initial quiz for any subject/subtopic."""
    # Validate that the subject/subtopic exists
    if not data_loader.validate_subject_subtopic(subject, subtopic):
        return f"Error: Subject '{subject}' with subtopic '{subtopic}' not found.", 404

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
        admin_override=session.get("admin_override", False),
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

    # Get allowed tags for the current subject
    allowed_topic_tags = get_subject_tags(current_subject)
    allowed_tags_str = json.dumps(allowed_topic_tags)

    prompt = (
        "You are analyzing a student's quiz submission which includes multiple choice, fill-in-the-blank, and coding questions.\n"
        "Based on the incorrect answers and their submitted code, identify the concepts they are weak in.\n"
        f"You **MUST** choose the weak concepts from this predefined list ONLY: {allowed_tags_str}\n\n"
        "For coding questions marked 'For AI Review', evaluate if the student's code:\n"
        "1. Correctly solves the problem\n"
        "2. Uses appropriate syntax and conventions\n"
        "3. Demonstrates understanding of the underlying concepts\n"
        "Compare their code with the provided sample solution.\n\n"
        "Provide your analysis as a single JSON object with two keys:\n"
        ' - "detailed_feedback": (string) Your textual analysis, including specific feedback on coding attempts, what they did well, and areas for improvement.\n'
        ' - "weak_concept_tags": (JSON list of strings) The list of weak concepts from the ALLOWED TAGS list. If there are no weaknesses, provide an empty list `[]`.\n\n'
        "Here is the student's submission:\n"
        "--- START OF SUBMISSION ---\n"
        f"{full_submission_text}"
        "--- END OF SUBMISSION ---\n"
    )

    ai_response_content = call_openai_api(
        prompt,
        system_message,
        model="gpt-4",
        max_tokens=1500,  # Increased tokens for more detailed feedback
        expect_json_output=True,
    )

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
        validated_weak_topics = [
            topic for topic in weak_topics if topic in allowed_topic_tags
        ]

        # Store weak topics with subject/subtopic prefix
        session[get_session_key(current_subject, current_subtopic, "weak_topics")] = (
            validated_weak_topics
        )
        app.logger.info(
            f"AI identified weak topics for {current_subject}/{current_subtopic}: {validated_weak_topics}"
        )

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


@app.route("/api/recommend_videos", methods=["GET"])
def recommend_videos_api():
    weak_topics_str = request.args.get("topics", "")

    if not weak_topics_str:
        return (
            jsonify({"error": "No weak topics provided for video recommendation"}),
            400,
        )

    weak_topics_list = [
        topic.strip().lower() for topic in weak_topics_str.split(",") if topic.strip()
    ]
    if not weak_topics_list:
        app.logger.info("Empty list of weak topics received for video recommendation.")
        return jsonify({"recommended_video_keys": []})

    # Use the legacy VIDEO_DATA structure for compatibility
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

    video_data_for_prompt = ""
    for key, video_info in VIDEO_DATA.items():
        video_data_for_prompt += f"Video Key: \"{key}\"\nTitle: \"{video_info['title']}\"\nDescription: \"{video_info['description']}\"\n\n"

    system_message = "You assist by recommending relevant educational video keys based on topic keywords and video descriptions. Output only a JSON list of video keys."
    prompt = (
        "Based on the following list of weak programming concepts a student has, "
        "and the provided list of available video materials (with their keys, titles, and descriptions), "
        "identify which video *keys* are most relevant for the student to review. "
        "Focus solely on matching the weak concepts to the video descriptions.\n\n"
        f"Weak Concepts:\n{', '.join(weak_topics_list)}\n\n"
        "Available Video Materials:\n"
        f"{video_data_for_prompt}\n"
        'Return your answer ONLY as a JSON formatted list of unique Video Keys. For example: ["key1", "key2"]. '
        "If no videos seem relevant for a particular concept, do not force a recommendation. "
        "If multiple videos seem relevant for the same concept, you can include all of them."
    )

    ai_response_content = call_openai_api(
        prompt,
        system_message,
        model="gpt-3.5-turbo",
        max_tokens=250,
        expect_json_output=True,
    )

    recommended_keys = None
    if ai_response_content:
        try:
            parsed_data = json.loads(ai_response_content)
            if isinstance(parsed_data, list):
                recommended_keys = parsed_data
            elif isinstance(parsed_data, dict):  # Handle if AI wraps in an object
                for key_in_dict in ["recommended_video_keys", "video_keys", "keys"]:
                    if isinstance(parsed_data.get(key_in_dict), list):
                        recommended_keys = parsed_data[key_in_dict]
                        break
            if (
                recommended_keys is None
            ):  # If direct parsing and common dict keys failed
                recommended_keys = parse_ai_json_from_text(
                    ai_response_content, expected_type_is_list=True
                )
        except json.JSONDecodeError:
            recommended_keys = parse_ai_json_from_text(
                ai_response_content, expected_type_is_list=True
            )

        if recommended_keys is not None and isinstance(recommended_keys, list):
            valid_recommended_keys = sorted(
                list(
                    set(
                        key
                        for key in recommended_keys
                        if key in VIDEO_DATA and isinstance(key, str)
                    )
                )
            )

            # Get current subject/subtopic from session for storage key
            current_subject = session.get("current_subject", "python")
            current_subtopic = session.get("current_subtopic", "functions")
            session[
                get_session_key(
                    current_subject,
                    current_subtopic,
                    "recommended_videos_for_weak_topics",
                )
            ] = valid_recommended_keys

            return jsonify({"recommended_video_keys": valid_recommended_keys})
        else:
            app.logger.error(
                f"Could not parse valid JSON list of video keys from AI (recommend_videos). Raw AI response: {ai_response_content}"
            )

    # Store empty result with prefix
    current_subject = session.get("current_subject", "python")
    current_subtopic = session.get("current_subtopic", "functions")
    session[
        get_session_key(
            current_subject, current_subtopic, "recommended_videos_for_weak_topics"
        )
    ] = []

    return (
        jsonify(
            {
                "error": "AI did not return valid video recommendations",
                "details": ai_response_content or "No response from AI",
            }
        ),
        500,
    )


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

    # Store the selected questions with subject/subtopic prefixes
    session[
        get_session_key(
            current_subject, current_subtopic, "current_remedial_quiz_questions"
        )
    ] = remedial_questions
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
        "quiz.html",
        questions=remedial_questions,
        quiz_title=quiz_title,
        admin_override=session.get("admin_override", False),
    )


@app.route("/results")
def show_results_page():
    quiz_gen_error = session.pop("quiz_generation_error", None)

    # Get current subject/subtopic from session - NO DEFAULTS TO PREVENT WRONG CONTEXT
    current_subject = session.get("current_subject")
    current_subtopic = session.get("current_subtopic")

    # If no session context, redirect to subject selection
    if not current_subject or not current_subtopic:
        app.logger.warning("No subject/subtopic context in session for results page")
        return redirect(url_for("subject_selection"))

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

    # Try to get lesson plans from the new system
    try:
        lesson_plans = get_lesson_plans(current_subject, current_subtopic)
    except Exception:
        lesson_plans = {}

    return render_template(
        "results.html",
        quiz_generation_error=quiz_gen_error,
        VIDEO_DATA=VIDEO_DATA,
        LESSON_PLANS=lesson_plans,
        current_subject=current_subject,
        current_subtopic=current_subtopic,
    )


#  ADMIN PANEL ROUTES


@app.route("/admin")
@app.route("/admin/")
def admin_dashboard():
    """Admin dashboard overview."""
    try:
        # Use auto-discovery instead of subjects.json
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
        # Use auto-discovery instead of subjects.json
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

            # Check if subject already exists by checking if directory exists
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
                "subject_info": {
                    "name": subject_name,
                    "description": description,
                    "icon": icon,
                    "color": color,
                },
                "subtopics": {},
                "allowed_tags": [],
            }

            config_path = os.path.join(subject_dir, "subject_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(subject_config, f, indent=2)

            # Clear cache to refresh subject list
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
        subject_config = data_loader.load_subject_config(subject)
        subject_info = data_loader.load_subject_info(subject)

        if not subject_config or not subject_info:
            return f"Subject '{subject}' not found", 404

        # Combine config and info into the expected structure
        config = {
            "subject_info": subject_info,
            "subtopics": subject_config.get("subtopics", {}),
            "allowed_tags": subject_config.get("allowed_tags", []),
        }

        return render_template(
            "admin/edit_subject.html", subject=subject, config=config
        )
    except Exception as e:
        app.logger.error(f"Error loading subject editor for {subject}: {e}")
        return f"Error: {e}", 500


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
        # Check if subject exists by checking directory
        subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject)
        if not os.path.exists(subject_dir):
            return jsonify({"error": "Subject not found"}), 404

        # Remove subject directory and all its contents
        shutil.rmtree(subject_dir)
        app.logger.info(f"Removed subject directory: {subject_dir}")

        # Clear cache to refresh subject list
        data_loader.clear_cache()

        return jsonify(
            {"success": True, "message": f"Subject '{subject}' deleted successfully"}
        )

    except Exception as e:
        app.logger.error(f"Error deleting subject {subject}: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ADMIN - OVERRIDE FUNCTIONALITY
# ============================================================================


@app.route("/admin/toggle-override", methods=["GET", "POST"])
def admin_toggle_override():
    """Toggle or check admin override status for debugging/testing."""
    try:
        if request.method == "GET":
            # Return current override status
            admin_override = session.get("admin_override", False)
            return jsonify({"success": True, "admin_override": admin_override})

        elif request.method == "POST":
            # Toggle the override status
            current_status = session.get("admin_override", False)
            new_status = not current_status
            session["admin_override"] = new_status

            message = f"Admin override {'enabled' if new_status else 'disabled'}"
            app.logger.info(f"Admin override toggled: {message}")

            return jsonify(
                {"success": True, "admin_override": new_status, "message": message}
            )

    except Exception as e:
        app.logger.error(f"Error in admin override toggle: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ADMIN - LESSONS MANAGEMENT ROUTES
# ============================================================================


def get_all_lessons():
    """Get all lessons across all subjects and subtopics."""
    lessons_data = []

    try:
        # Use auto-discovery instead of subjects.json
        subjects = data_loader.discover_subjects()

        for subject_id, subject_info in subjects.items():
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
                                    "subject_name": subject_info.get(
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

            return True
        return False
    except Exception as e:
        app.logger.error(f"Error deleting lesson {lesson_id}: {e}")
        return False


@app.route("/admin/lessons")
def admin_lessons():
    """List all lessons across all subjects."""
    lessons = get_all_lessons()

    # Use auto-discovery for subjects dropdown
    subjects = data_loader.discover_subjects()

    return render_template("admin/lessons.html", lessons=lessons, subjects=subjects)


@app.route("/admin/lessons/create", methods=["GET", "POST"])
def admin_create_lesson():
    """Create a new lesson."""
    if request.method == "POST":
        try:
            data = request.json
            subject = data.get("subject")
            subtopic = data.get("subtopic")
            lesson_id = data.get("id", "").lower().replace(" ", "_")
            lesson_title = data.get("title", "")
            video_id = data.get("videoId", "")
            content = data.get("content", [])
            tags = data.get("tags", [])

            if not all([subject, subtopic, lesson_id, lesson_title]):
                return (
                    jsonify(
                        {
                            "error": "Subject, subtopic, lesson ID, and title are required"
                        }
                    ),
                    400,
                )

            # Validate subject and subtopic exist
            if not data_loader.validate_subject_subtopic(subject, subtopic):
                return (
                    jsonify(
                        {
                            "error": f"Subject '{subject}' with subtopic '{subtopic}' not found"
                        }
                    ),
                    404,
                )

            # Check if lesson already exists
            existing_lessons = get_lesson_plans(subject, subtopic)
            if lesson_id in existing_lessons:
                return jsonify({"error": "Lesson already exists"}), 400

            # Create lesson data
            lesson_data = {
                "title": lesson_title,
                "videoId": video_id,
                "content": content,
                "tags": tags,
                "created_date": "2025-01-01",
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

    # GET request - show form
    try:
        # Use auto-discovery for subjects dropdown
        subjects = data_loader.discover_subjects()
        return render_template("admin/create_lesson.html", subjects=subjects)
    except Exception as e:
        app.logger.error(f"Error loading lesson creation form: {e}")
        return f"Error: {e}", 500


@app.route(
    "/admin/lessons/<subject>/<subtopic>/<lesson_id>/edit", methods=["GET", "POST"]
)
def admin_edit_lesson(subject, subtopic, lesson_id):
    """Edit an existing lesson."""
    if request.method == "POST":
        try:
            data = request.json
            lesson_title = data.get("title", "")
            video_id = data.get("videoId", "")
            content = data.get("content", [])
            tags = data.get("tags", [])

            if not lesson_title:
                return jsonify({"error": "Lesson title is required"}), 400

            # Update lesson data
            lesson_data = {
                "title": lesson_title,
                "videoId": video_id,
                "content": content,
                "tags": tags,
                "updated_date": "2025-01-01",
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
        # Load existing lesson
        lessons = get_lesson_plans(subject, subtopic)
        if lesson_id not in lessons:
            return f"Lesson '{lesson_id}' not found", 404

        lesson_data = lessons[lesson_id]

        # Get subjects for context
        subjects = data_loader.discover_subjects()

        # Load subtopics for the current subject (needed for edit mode)
        subject_config = data_loader.load_subject_config(subject)
        subject_subtopics = (
            subject_config.get("subtopics", {}) if subject_config else {}
        )

        return render_template(
            "admin/create_lesson.html",
            subjects=subjects,
            edit_mode=True,
            subject=subject,
            subtopic=subtopic,
            lesson_id=lesson_id,
            lesson_data=lesson_data,
            subject_subtopics=subject_subtopics,
        )
    except Exception as e:
        app.logger.error(f"Error loading lesson editor: {e}")
        return f"Error: {e}", 500


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


@app.route("/admin/subtopics")
def admin_subtopics():
    """Manage subtopics across all subjects."""
    try:
        # Use auto-discovery instead of subjects.json
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
                    allowed_tags = config.get(
                        "allowed_tags", config.get("allowed_keywords", [])
                    )

                    subjects[subject_id]["subtopics"] = config_subtopics
                    subjects[subject_id]["allowed_tags"] = allowed_tags

                except Exception as e:
                    app.logger.error(
                        f"Error reading subject config for {subject_id}: {e}"
                    )
                    subjects[subject_id]["subtopics"] = {}
                    subjects[subject_id]["allowed_tags"] = []
            else:
                subjects[subject_id]["subtopics"] = {}
                subjects[subject_id]["allowed_tags"] = []

        return render_template("admin/subtopics.html", subjects=subjects)

    except Exception as e:
        app.logger.error(f"Error loading subtopics: {e}")
        return render_template("admin/subtopics.html", subjects={})


@app.route("/admin/questions")
def admin_questions():
    """Questions management page."""
    try:
        # Use auto-discovery instead of subjects.json
        subjects_data = {}
        stats = {
            "total_initial_questions": 0,
            "total_pool_questions": 0,
            "total_subtopics": 0,
            "subtopics_without_questions": 0,
        }

        # Discover subjects using auto-discoverythe subtopic
        discovered_subjects = data_loader.discover_subjects()

        for subject_id, subject_info in discovered_subjects.items():
            # Load subject config to get subtopics
            subject_config = data_loader.load_subject_config(subject_id)
            if subject_config and "subtopics" in subject_config:
                subject_data = {
                    "name": subject_info.get("name", subject_id),
                    "description": subject_info.get("description", ""),
                    "subtopics": {},
                }

                for subtopic_id, subtopic_data in subject_config["subtopics"].items():
                    # Load quiz data and question pool to get counts
                    quiz_data = data_loader.load_quiz_data(subject_id, subtopic_id)
                    pool_data = data_loader.get_question_pool_questions(
                        subject_id, subtopic_id
                    )

                    quiz_count = len(quiz_data.get("questions", [])) if quiz_data else 0
                    pool_count = len(pool_data) if pool_data else 0

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
    """Quiz editor page for a specific subject/subtopic."""
    try:
        # Validate subject/subtopic exists
        if not data_loader.validate_subject_subtopic(subject, subtopic):
            return f"Subject '{subject}' with subtopic '{subtopic}' not found", 404

        # Load quiz data and question pool
        quiz_data = data_loader.load_quiz_data(subject, subtopic)
        pool_questions = data_loader.get_question_pool_questions(subject, subtopic)

        # Format question pool to match template expectations
        question_pool = {"questions": pool_questions}

        # Get subject config for tags
        subject_config = data_loader.load_subject_config(subject)
        allowed_tags = (
            subject_config.get(
                "allowed_tags", subject_config.get("allowed_keywords", [])
            )
            if subject_config
            else []
        )

        return render_template(
            "admin/quiz_editor.html",
            subject=subject,
            subtopic=subtopic,
            quiz_data=quiz_data,
            question_pool=question_pool,
            allowed_tags=allowed_tags,
        )

    except Exception as e:
        app.logger.error(f"Error loading quiz editor: {e}")
        return f"Error: {e}", 500


@app.route("/admin/quiz/<subject>/<subtopic>/initial", methods=["GET", "POST"])
def admin_quiz_initial(subject, subtopic):
    """Manage initial quiz questions."""
    if request.method == "GET":
        try:
            quiz_data = data_loader.load_quiz_data(subject, subtopic)
            return jsonify(quiz_data if quiz_data else {"questions": []})
        except Exception as e:
            app.logger.error(f"Error loading initial quiz data: {e}")
            return jsonify({"error": str(e)}), 500

    elif request.method == "POST":
        try:
            data = request.json
            questions = data.get("questions", [])

            # Create quiz data structure
            quiz_data = {
                "quiz_title": f"{subject.title()} - {subtopic.title()} Quiz",
                "questions": questions,
                "updated_date": "2025-01-01",
            }

            # Save to file
            quiz_file_path = os.path.join(
                DATA_ROOT_PATH, "subjects", subject, subtopic, "quiz_data.json"
            )

            # Ensure directory exists
            os.makedirs(os.path.dirname(quiz_file_path), exist_ok=True)

            with open(quiz_file_path, "w", encoding="utf-8") as f:
                json.dump(quiz_data, f, indent=2)

            return jsonify(
                {"success": True, "message": "Initial quiz updated successfully"}
            )

        except Exception as e:
            app.logger.error(f"Error updating initial quiz: {e}")
            return jsonify({"error": str(e)}), 500


@app.route("/admin/quiz/<subject>/<subtopic>/pool", methods=["GET", "POST"])
def admin_quiz_pool(subject, subtopic):
    """Manage question pool for remedial quizzes."""
    if request.method == "GET":
        try:
            pool_data = data_loader.get_question_pool_questions(subject, subtopic)
            return jsonify({"questions": pool_data if pool_data else []})
        except Exception as e:
            app.logger.error(f"Error loading question pool: {e}")
            return jsonify({"error": str(e)}), 500

    elif request.method == "POST":
        try:
            data = request.json
            questions = data.get("questions", [])

            # Create question pool structure
            pool_data = {
                "pool_title": f"{subject.title()} - {subtopic.title()} Question Pool",
                "questions": questions,
                "updated_date": "2025-01-01",
            }

            # Save to file
            pool_file_path = os.path.join(
                DATA_ROOT_PATH, "subjects", subject, subtopic, "question_pool.json"
            )

            # Ensure directory exists
            os.makedirs(os.path.dirname(pool_file_path), exist_ok=True)

            with open(pool_file_path, "w", encoding="utf-8") as f:
                json.dump(pool_data, f, indent=2)

            return jsonify(
                {"success": True, "message": "Question pool updated successfully"}
            )

        except Exception as e:
            app.logger.error(f"Error updating question pool: {e}")
            return jsonify({"error": str(e)}), 500


@app.route("/api/subjects/<subject>/tags")
def api_get_subject_tags(subject):
    """API endpoint to get available tags for a subject."""
    try:
        tags = get_subject_tags(subject)

        return jsonify({"success": True, "tags": tags, "count": len(tags)})

    except Exception as e:
        app.logger.error(f"Error getting tags for {subject}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/subjects/<subject>/subtopics")
def api_get_subtopics(subject):
    """API endpoint to get subtopics for a subject."""
    try:
        subject_config = data_loader.load_subject_config(subject)
        if not subject_config:
            return jsonify({"error": "Subject not found"}), 404

        subtopics = subject_config.get("subtopics", {})

        # Return the subtopics in the format expected by the frontend JavaScript
        # The frontend expects an object with subtopic IDs as keys
        return jsonify({"subtopics": subtopics})

    except Exception as e:
        app.logger.error(f"Error getting subtopics for {subject}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/lessons/find-by-tags", methods=["POST"])
def api_find_lessons_by_tags():
    """API endpoint to find lessons matching specific tags."""
    try:
        data = request.json
        subject = data.get("subject")
        target_tags = data.get("tags", [])

        if not subject:
            return jsonify({"error": "Subject is required"}), 400

        if not target_tags:
            return jsonify({"lessons": []})

        # Use the DataLoader method to find matching lessons
        matching_lessons = data_loader.find_lessons_by_tags(subject, target_tags)

        app.logger.info(
            f"Found {len(matching_lessons)} lessons for subject '{subject}' with tags {target_tags}"
        )

        return jsonify(
            {
                "lessons": matching_lessons,
                "count": len(matching_lessons),
                "searched_tags": target_tags,
            }
        )

    except Exception as e:
        app.logger.error(f"Error finding lessons by tags: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/admin/export")
def admin_export():
    """Export/Import functionality placeholder."""
    return render_template("admin/export.html")


@app.route("/admin/clear-cache", methods=["POST"])
def admin_clear_cache():
    """Clear the DataLoader cache."""
    try:
        # Clear the DataLoader cache
        data_loader._cache.clear()

        app.logger.info("DataLoader cache cleared successfully")
        return jsonify(
            {
                "success": True,
                "message": "Cache cleared successfully! All data will be reloaded fresh.",
            }
        )

    except Exception as e:
        app.logger.error(f"Error clearing cache: {e}")
        return (
            jsonify({"success": False, "error": f"Failed to clear cache: {str(e)}"}),
            500,
        )


@app.route("/admin/migrate-tags", methods=["POST"])
def admin_migrate_tags():
    """Migrate all subjects from keywords to tags format."""
    try:
        # Perform migration for all subjects
        results = data_loader.migrate_all_subjects_tags()

        successful_migrations = [
            subject for subject, success in results.items() if success
        ]
        failed_migrations = [
            subject for subject, success in results.items() if not success
        ]

        # Clear cache to ensure new data is loaded
        data_loader._cache.clear()

        message = f"Migration completed! Successfully migrated {len(successful_migrations)} subjects."
        if failed_migrations:
            message += f" Failed to migrate: {', '.join(failed_migrations)}"

        app.logger.info(f"Tag migration results: {results}")

        return jsonify(
            {
                "success": True,
                "message": message,
                "results": results,
                "successful_count": len(successful_migrations),
                "failed_count": len(failed_migrations),
            }
        )

    except Exception as e:
        app.logger.error(f"Error during tag migration: {e}")
        return (
            jsonify({"success": False, "error": f"Failed to migrate tags: {str(e)}"}),
            500,
        )


@app.route("/api/lessons/<subject>/<subtopic>/<lesson_id>")
def api_get_lesson(subject, subtopic, lesson_id):
    """Return a specific lesson by subject/subtopic/lesson_id."""
    try:
        # Validate subject/subtopic exists
        if not data_loader.validate_subject_subtopic(subject, subtopic):
            return jsonify({"error": "Subject or subtopic not found"}), 404

        lesson_plans = data_loader.load_lesson_plans(subject, subtopic)
        lessons = lesson_plans.get("lessons", {}) if lesson_plans else {}

        if lesson_id in lessons:
            return jsonify({"lesson": lessons[lesson_id]})

        # Fallback: try resolving by case-insensitive title match
        for key, value in lessons.items():
            title = value.get("title")
            if title and title.lower() == lesson_id.lower():
                return jsonify({"lesson": value})

        return jsonify({"error": "Lesson not found"}), 404
    except Exception as e:
        app.logger.error(f"Error fetching lesson {subject}/{subtopic}/{lesson_id}: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY environment variable not set. AI features will not work."
        )

    # Validate that we have the required data structure
    if not data_loader.validate_subject_subtopic("python", "functions"):
        print(
            "ERROR: Python functions data not found. Check data/subjects/python/functions/ directory."
        )

    app.run(debug=True)
