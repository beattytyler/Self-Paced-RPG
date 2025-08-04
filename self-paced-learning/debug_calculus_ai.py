#!/usr/bin/env python3
"""
Debug script to test AI analysis for calculus quiz
Simulates getting all calculus questions wrong and checks AI analysis
"""

import json
import os
from utils.data_loader import DataLoader


def simulate_calculus_quiz():
    """Simulate taking calculus quiz and getting all questions wrong"""

    # Initialize data loader
    data_loader = DataLoader("data")

    # Load calculus quiz data
    quiz_data = data_loader.load_quiz_data("calculus", "integrals")
    if not quiz_data:
        print("ERROR: Could not load calculus quiz data")
        return

    print("=== CALCULUS QUIZ SIMULATION ===")
    print(f"Quiz title: {quiz_data.get('quiz_title', 'N/A')}")
    print(f"Number of questions: {len(quiz_data.get('questions', []))}")
    print()

    # Simulate getting all questions wrong
    user_answers = []
    correct_answers = []

    for i, question in enumerate(quiz_data.get("questions", [])):
        print(f"Question {i+1}: {question.get('question', 'N/A')}")
        print(f"  Type: {question.get('type', 'N/A')}")
        print(f"  Tags: {question.get('tags', [])}")

        if question.get("type") == "multiple_choice":
            correct_idx = question.get("answer_index", 0)
            # Choose wrong answer (any index that's not correct)
            wrong_idx = 1 if correct_idx == 0 else 0
            user_answers.append(wrong_idx)
            correct_answers.append(correct_idx)
            print(f"  Correct answer: {correct_idx}")
            print(f"  User chose: {wrong_idx} (WRONG)")

        print()

    # Calculate score
    correct_count = sum(1 for u, c in zip(user_answers, correct_answers) if u == c)
    score = correct_count / len(user_answers) if user_answers else 0

    print(f"=== QUIZ RESULTS ===")
    print(f"Score: {correct_count}/{len(user_answers)} ({score:.1%})")
    print(f"Failed questions: {len(user_answers) - correct_count}")
    print()

    # Now simulate what the AI should analyze
    failed_questions = []
    for i, (user_ans, correct_ans, question) in enumerate(
        zip(user_answers, correct_answers, quiz_data.get("questions", []))
    ):
        if user_ans != correct_ans:
            failed_questions.append(
                {
                    "question_number": i + 1,
                    "question": question.get("question", ""),
                    "tags": question.get("tags", []),
                    "user_answer": user_ans,
                    "correct_answer": correct_ans,
                }
            )

    print("=== FAILED QUESTIONS ANALYSIS ===")
    for fq in failed_questions:
        print(f"Question {fq['question_number']}: {fq['question'][:50]}...")
        print(f"  Tags: {fq['tags']}")
    print()

    # Check allowed keywords for calculus
    subject_config = data_loader.load_subject_config("calculus")
    if subject_config:
        allowed_keywords = subject_config.get("allowed_keywords", [])
        print("=== CALCULUS ALLOWED KEYWORDS ===")
        print(allowed_keywords)
        print()

        # Extract all tags from failed questions
        all_failed_tags = []
        for fq in failed_questions:
            all_failed_tags.extend(fq["tags"])

        # Remove duplicates
        unique_failed_tags = list(set(all_failed_tags))
        print(f"=== UNIQUE TAGS FROM FAILED QUESTIONS ===")
        print(unique_failed_tags)
        print()

        # Check which tags are valid
        valid_weak_topics = [
            tag for tag in unique_failed_tags if tag in allowed_keywords
        ]
        invalid_weak_topics = [
            tag for tag in unique_failed_tags if tag not in allowed_keywords
        ]

        print(f"=== VALIDATION RESULTS ===")
        print(f"Valid weak topics: {valid_weak_topics}")
        print(f"Invalid weak topics: {invalid_weak_topics}")
        print()

        # Test lesson search
        if valid_weak_topics:
            matching_lessons = data_loader.find_lessons_by_tags(
                "calculus", valid_weak_topics
            )
            print(f"=== LESSON SEARCH RESULTS ===")
            print(f"Searching for lessons with topics: {valid_weak_topics}")
            print(f"Found {len(matching_lessons)} lessons:")
            for lesson in matching_lessons:
                print(f"  - {lesson['title']} (tags: {lesson['tags']})")
        else:
            print("=== NO VALID WEAK TOPICS TO SEARCH FOR LESSONS ===")

    else:
        print("ERROR: Could not load calculus subject config")


if __name__ == "__main__":
    simulate_calculus_quiz()
