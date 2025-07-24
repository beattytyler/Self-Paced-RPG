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



#  Main Application Routes 
@app.route("/")
def subject_selection():
    """New home page showing all available subjects."""
    try:
        # Load subjects from subjects.json
        subjects_path = os.path.join(DATA_ROOT_PATH, "subjects.json")
        if os.path.exists(subjects_path):
            with open(subjects_path, 'r', encoding='utf-8') as f:
                subjects_data = json.load(f)
                subjects = subjects_data.get("subjects", {})
        else:
            # Fallback to default Python subject if file doesn't exist
            subjects = {
                "python": {
                    "name": "Python Programming",
                    "description": "Master Python from basics to advanced topics including functions, loops, data structures, and more.",
                    "icon": "fab fa-python",
                    "color": "#3776ab",
                    "status": "active",
                    "subtopic_count": 6
                }
            }
        
        return render_template("subject_selection.html", subjects=subjects)
    except Exception as e:
        app.logger.error(f"Error loading subject selection: {e}")
        # Fallback to legacy index if there's an error
        return redirect(url_for('python_subject_page'))

@app.route("/subjects/<subject>")
def subject_page(subject):
    """Display subtopics for a specific subject."""
    try:
        # Load subject configuration
        subject_config = data_loader.load_subject_config(subject)
        if not subject_config:
            app.logger.error(f"Subject config not found for: {subject}")
            return redirect(url_for('subject_selection'))
        
        subject_info = subject_config.get("subject_info", {})
        subtopics = subject_config.get("subtopics", {})
        
        # Sort subtopics by order
        sorted_subtopics = dict(sorted(subtopics.items(), key=lambda x: x[1].get('order', 999)))
        
        return render_template("python_subject.html", 
                             subject=subject,
                             subject_info=subject_info, 
                             subtopics=sorted_subtopics)
    except Exception as e:
        app.logger.error(f"Error loading subject page for {subject}: {e}")
        return redirect(url_for('subject_selection'))

@app.route("/legacy")
def legacy_index():
    """Legacy route - redirects to Python subject page."""
    return redirect(url_for('subject_page', subject='python'))

@app.route("/python")  
def python_subject_page():
    """Direct route to Python subject - for backward compatibility."""
    return redirect(url_for('subject_page', subject='python'))


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
    session[get_session_key(subject, subtopic, "questions_served_for_analysis")] = quiz_questions
    session["current_subject"] = subject
    session["current_subtopic"] = subtopic
    
    return render_template(
        "quiz.html", questions=quiz_questions, quiz_title=quiz_title
    )

# Legacy route for backward compatibility
@app.route("/quiz/functions")
def quiz_functions_page():
    """Legacy route - redirects to the new structure."""
    return redirect(url_for('quiz_page', subject='python', subtopic='functions'))
@app.route("/analyze", methods=["POST"])
def analyze_quiz():
    user_submitted_answers = request.json.get("answers", {})
    
    # Get current subject/subtopic from session
    current_subject = session.get("current_subject")
    current_subtopic = session.get("current_subtopic")
    
    if not current_subject or not current_subtopic:
        app.logger.error("No current subject/subtopic found in session for analysis.")
        return jsonify({"feedback": "Error: Quiz session data not found.", "weak_topics": []}), 400
    
    # Get questions that were served for analysis using prefixed session key
    questions_for_analysis = session.get(get_session_key(current_subject, current_subtopic, "questions_served_for_analysis"), [])

    if not questions_for_analysis:
        app.logger.error("No questions found in session for analysis.")
        return jsonify({"feedback": "Error: Quiz session data not found.", "weak_topics": []}), 400

    submission_details_list = []
    correct_answers = 0
    total_questions = len(questions_for_analysis)
    
    for i, q_data in enumerate(questions_for_analysis):
        user_answer = user_submitted_answers.get(f"q{i}", "[No answer provided]")
        question_type = q_data.get("type", "multiple_choice")  # Default to multiple_choice for backward compatibility
        status = "Incorrect"  # Default status

        detail = f"Question {i+1} (Type: {question_type}): {q_data.get('question', 'N/A')}\n"
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
            correct_answers_list = [ans.strip().lower() for ans in correct_answer_text.split(",")]
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
            sample_solution = q_data.get('sample_solution', '')
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
        model="gpt-4",
        max_tokens=1500, # Increased tokens for more detailed feedback
        expect_json_output=True
    )

    if not ai_response_content:
        return jsonify({"feedback": "Error: Could not get analysis from AI.", "weak_topics": []}), 500

    json_match = re.search(r'\{[\s\S]*\}', ai_response_content)
    if not json_match:
        app.logger.error(f"Could not find JSON in AI response.\nResponse was: {ai_response_content}")
        return jsonify({"feedback": "Error: The analysis response did not contain a valid JSON object.", "weak_topics": []}), 500

    try:
        parsed_ai_response = json.loads(json_match.group(0))
        feedback = parsed_ai_response.get("detailed_feedback", "No detailed feedback provided.")
        weak_topics = parsed_ai_response.get("weak_concept_tags", [])
        validated_weak_topics = [topic for topic in weak_topics if topic in allowed_topic_keywords]
        
        # Store weak topics with subject/subtopic prefix
        session[get_session_key(current_subject, current_subtopic, "weak_topics")] = validated_weak_topics
        app.logger.info(f"AI identified weak topics for {current_subject}/{current_subtopic}: {validated_weak_topics}")

        # Calculate score percentage
        score_percentage = round((correct_answers / total_questions) * 100) if total_questions > 0 else 0

        return jsonify({
            "feedback": feedback,
            "weak_topics": validated_weak_topics,
            "score": {
                "correct": correct_answers,
                "total": total_questions,
                "percentage": score_percentage
            }
        })

    except json.JSONDecodeError as e:
        app.logger.error(f"Failed to parse extracted AI JSON response: {e}\nExtracted text was: {json_match.group(0)}")
        return jsonify({"feedback": "Error: The analysis response format was invalid.", "weak_topics": []}), 500

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
            session[get_session_key(current_subject, current_subtopic, "recommended_videos_for_weak_topics")] = valid_recommended_keys
            
            return jsonify({"recommended_video_keys": valid_recommended_keys})
        else:
            app.logger.error(
                f"Could not parse valid JSON list of video keys from AI (recommend_videos). Raw AI response: {ai_response_content}"
            )

    # Store empty result with prefix
    current_subject = session.get("current_subject", "python")
    current_subtopic = session.get("current_subtopic", "functions")
    session[get_session_key(current_subject, current_subtopic, "recommended_videos_for_weak_topics")] = []
    
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
        app.logger.error("No current subject/subtopic found in session for remedial quiz generation.")
        session["quiz_generation_error"] = "Session error: Please take the main quiz first."
        return redirect(url_for("show_results_page"))
    
    # Get weak topics with subject/subtopic prefix
    weak_topics = session.get(get_session_key(current_subject, current_subtopic, "weak_topics"), [])
    
    if not weak_topics:
        app.logger.info(f"No weak topics in session for {current_subject}/{current_subtopic}; cannot generate remedial quiz.")
        session["quiz_generation_error"] = "You've mastered all identified topics! No remedial quiz needed."
        return redirect(url_for("show_results_page"))

    # Get question pool for current subject/subtopic
    question_pool = get_question_pool(current_subject, current_subtopic)

    # Select questions from the pool that match the weak topics
    remedial_questions = []
    selected_questions_set = set() # To avoid duplicate questions

    app.logger.info(f"Filtering question pool for weak topics in {current_subject}/{current_subtopic}: {weak_topics}")

    for question in question_pool:
        # Check if any of the question's tags are in the user's weak topics
        question_tags = set(question.get("tags", []))
        if not question_tags.isdisjoint(weak_topics):
            # Use the question's text as a unique identifier to avoid duplicates
            if question['question'] not in selected_questions_set:
                remedial_questions.append(question)
                selected_questions_set.add(question['question'])

    if not remedial_questions:
        app.logger.warning(f"No questions found in question pool for topics: {weak_topics} in {current_subject}/{current_subtopic}")
        session["quiz_generation_error"] = "We couldn't find specific follow-up questions for your weak topics. Please review the materials and try the main quiz again."
        return redirect(url_for("show_results_page"))

    # Store the selected questions with subject/subtopic prefixes
    session[get_session_key(current_subject, current_subtopic, "current_remedial_quiz_questions")] = remedial_questions
    session[get_session_key(current_subject, current_subtopic, "questions_served_for_analysis")] = remedial_questions
    session[get_session_key(current_subject, current_subtopic, "current_quiz_type")] = "remedial"
    session[get_session_key(current_subject, current_subtopic, "topics_for_current_remedial_quiz")] = weak_topics

    app.logger.info(f"Selected {len(remedial_questions)} questions for the remedial quiz in {current_subject}/{current_subtopic}.")

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
    remedial_questions = session.get(get_session_key(current_subject, current_subtopic, "current_remedial_quiz_questions"), [])
    
    if not remedial_questions:
        app.logger.info(
            f"No remedial quiz in session for {current_subject}/{current_subtopic}, redirecting to results with error."
        )
        session["quiz_generation_error"] = (
            "No remedial quiz was available to take. Perhaps try again or review more."
        )
        return redirect(url_for("show_results_page"))

    quiz_title = "Remedial Quiz"
    targeted_topics = session.get(get_session_key(current_subject, current_subtopic, "topics_for_current_remedial_quiz"))
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
                    "description": video_info.get("description", "")
                }
        else:
            # Fallback to default Python topics if no video data
            VIDEO_DATA = {
                "functions": {
                    "title": "Python Functions Masterclass",
                    "url": "https://www.youtube.com/embed/kvO_nHnvPtQ?enablejsapi=1",
                    "description": "Master Python functions, parameters, return values, and scope."
                },
                "loops": {
                    "title": "Python Loops: For and While",
                    "url": "https://www.youtube.com/watch?v=94UHCEmprCY",
                    "description": "Learn how to automate repetitive tasks using for and while loops."
                }
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
        current_subtopic=current_subtopic
    )


#  ADMIN PANEL ROUTES 

@app.route("/admin")
def admin_dashboard():
    """Admin dashboard overview."""
    try:
        # Load subjects data
        subjects_path = os.path.join(DATA_ROOT_PATH, "subjects.json")
        if os.path.exists(subjects_path):
            with open(subjects_path, 'r', encoding='utf-8') as f:
                subjects_data = json.load(f)
                subjects = subjects_data.get("subjects", {})
        else:
            subjects = {}
        
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
            "total_questions": total_questions
        }
        
        return render_template("admin/dashboard.html", subjects=subjects, stats=stats)
    except Exception as e:
        app.logger.error(f"Error loading admin dashboard: {e}")
        return f"Error loading admin dashboard: {e}", 500

@app.route("/admin/subjects")
def admin_subjects():
    """Manage subjects."""
    try:
        subjects_path = os.path.join(DATA_ROOT_PATH, "subjects.json")
        if os.path.exists(subjects_path):
            with open(subjects_path, 'r', encoding='utf-8') as f:
                subjects_data = json.load(f)
                subjects = subjects_data.get("subjects", {})
        else:
            subjects = {}
            
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
            
            # Load existing subjects
            subjects_path = os.path.join(DATA_ROOT_PATH, "subjects.json")
            if os.path.exists(subjects_path):
                with open(subjects_path, 'r', encoding='utf-8') as f:
                    subjects_data = json.load(f)
            else:
                subjects_data = {"subjects": {}}
            
            # Check if subject already exists
            if subject_id in subjects_data["subjects"]:
                return jsonify({"error": "Subject already exists"}), 400
            
            # Add new subject
            subjects_data["subjects"][subject_id] = {
                "name": subject_name,
                "description": description,
                "icon": icon,
                "color": color,
                "status": "active",
                "created_date": "2025-01-01",
                "subtopic_count": 0
            }
            
            # Save subjects.json
            with open(subjects_path, 'w', encoding='utf-8') as f:
                json.dump(subjects_data, f, indent=2)
            
            # Create subject directory and config
            subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject_id)
            os.makedirs(subject_dir, exist_ok=True)
            
            subject_config = {
                "subject_info": {
                    "name": subject_name,
                    "description": description,
                    "icon": icon,
                    "color": color
                },
                "subtopics": {},
                "allowed_keywords": []
            }
            
            config_path = os.path.join(subject_dir, "subject_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(subject_config, f, indent=2)
            
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
        if not subject_config:
            return f"Subject '{subject}' not found", 404
            
        return render_template("admin/edit_subject.html", subject=subject, config=subject_config)
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
        
        return render_template("admin/edit_subtopic.html", 
                             subject=subject, 
                             subtopic=subtopic, 
                             subtopic_data=subtopic_data,
                             quiz_data=quiz_data,
                             lesson_plans=lesson_plans,
                             videos=videos)
    except Exception as e:
        app.logger.error(f"Error loading subtopic editor for {subject}/{subtopic}: {e}")
        return f"Error: {e}", 500

@app.route("/admin/subjects/<subject>/delete", methods=["DELETE"])
def admin_delete_subject(subject):
    """Delete a subject and all its associated data."""
    try:
        # Load existing subjects
        subjects_path = os.path.join(DATA_ROOT_PATH, "subjects.json")
        if not os.path.exists(subjects_path):
            return jsonify({"error": "Subjects file not found"}), 404
        
        with open(subjects_path, 'r', encoding='utf-8') as f:
            subjects_data = json.load(f)
        
        # Check if subject exists
        if subject not in subjects_data.get("subjects", {}):
            return jsonify({"error": "Subject not found"}), 404
        
        # Remove subject from subjects.json
        del subjects_data["subjects"][subject]
        
        # Save subjects.json
        with open(subjects_path, 'w', encoding='utf-8') as f:
            json.dump(subjects_data, f, indent=2)
        
        # Remove subject directory and all its contents
        subject_dir = os.path.join(DATA_ROOT_PATH, "subjects", subject)
        if os.path.exists(subject_dir):
            shutil.rmtree(subject_dir)
            app.logger.info(f"Removed subject directory: {subject_dir}")
        
        return jsonify({"success": True, "message": f"Subject '{subject}' deleted successfully"})
        
    except Exception as e:
        app.logger.error(f"Error deleting subject {subject}: {e}")
        return jsonify({"error": str(e)}), 500


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
