#!/usr/bin/env python3
"""
Quick test script to verify lesson tagging functionality.
Run this to test the new lesson tagging system.
"""

import os
import sys

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader


def test_lesson_tags():
    """Test the lesson tagging functionality."""
    print("=== Testing Lesson Tag Functionality ===\n")

    # Initialize data loader
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    loader = DataLoader(data_root)

    # Test 1: Load Python subject keywords
    print("1. Testing subject keywords loading...")
    keywords = loader.get_subject_keywords("python")
    print(f"   Python allowed keywords: {keywords}")
    print(f"   Found {len(keywords)} keywords\n")

    # Test 2: Test finding lessons by tags
    print("2. Testing lesson search by tags...")
    test_tags = ["python function basics", "syntax"]
    matching_lessons = loader.find_lessons_by_tags("python", test_tags)

    print(f"   Searching for lessons with tags: {test_tags}")
    print(f"   Found {len(matching_lessons)} matching lessons:")

    for lesson in matching_lessons:
        print(f"     - {lesson['title']} ({lesson['subject']}/{lesson['subtopic']})")
        print(f"       Tags: {lesson['tags']}")
        print(f"       Matching: {lesson['matching_tags']}")
        print()

    # Test 3: Test with non-existent tags
    print("3. Testing with non-existent tags...")
    fake_tags = ["nonexistent", "fake-tag"]
    no_matches = loader.find_lessons_by_tags("python", fake_tags)
    print(f"   Searching for lessons with tags: {fake_tags}")
    print(f"   Found {len(no_matches)} matching lessons (should be 0)\n")

    print("=== Test Complete ===")


if __name__ == "__main__":
    test_lesson_tags()
