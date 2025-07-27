#!/usr/bin/env python3
"""
Test script to show all lesson titles
"""
import sys
import os
import json

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from utils.data_loader import DataLoader


def show_all_lessons():
    """Show all lesson titles"""

    # Initialize DataLoader
    data_root = os.path.join(project_dir, "data")
    loader = DataLoader(data_root)

    print("📚 All lessons in python/functions:\n")

    # Load lesson plans
    lessons = loader.load_lesson_plans("python", "functions")
    if lessons:
        lesson_list = lessons.get("lessons", {})
        for i, (lesson_id, lesson_data) in enumerate(lesson_list.items(), 1):
            title = lesson_data.get("title", "No title")
            print(f"{i:2d}. {lesson_id}")
            print(f"    Title: {title}")
            print(f"    Video: {lesson_data.get('videoId', 'None')}")
            print(f"    Content blocks: {len(lesson_data.get('content', []))}")
            print()

        print(f"Total lessons: {len(lesson_list)}")

        # Check for your new lessons specifically
        if "test" in lesson_list:
            print("✅ Found your 'test' lesson!")
        if "python_syntax" in lesson_list:
            print("✅ Found your 'python_syntax' lesson!")

    else:
        print("❌ Failed to load lessons")


if __name__ == "__main__":
    show_all_lessons()
