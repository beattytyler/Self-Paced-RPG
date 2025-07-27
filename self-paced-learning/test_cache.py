#!/usr/bin/env python3
"""
Test script to verify cache clearing functionality
"""
import sys
import os
import json

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from utils.data_loader import DataLoader


def test_cache_clearing():
    """Test that cache clearing works properly"""

    # Initialize DataLoader
    data_root = os.path.join(project_dir, "data")
    loader = DataLoader(data_root)

    print("🧪 Testing cache clearing functionality...\n")

    # Load lesson plans (this will cache them)
    print("1. Loading lesson plans (caching them)...")
    lessons1 = loader.load_lesson_plans("python", "functions")
    if lessons1:
        print(f"   ✅ Loaded {len(lessons1.get('lessons', {}))} lessons")
        # Show some content to verify it contains "test"
        first_lesson = list(lessons1.get("lessons", {}).values())[0]
        title = first_lesson.get("title", "")
        print(f"   📖 First lesson title: {title}")
        has_test = "test" in title.lower()
        print(f"   🔍 Contains 'test': {has_test}")
    else:
        print("   ❌ Failed to load lessons")
        return

    print("\n2. Checking cache contents...")
    cache_size_before = len(loader._cache)
    print(f"   📦 Cache has {cache_size_before} entries")

    # Clear cache for specific subject/subtopic
    print("\n3. Clearing cache for python/functions...")
    loader.clear_cache_for_subject_subtopic("python", "functions")
    cache_size_after = len(loader._cache)
    print(f"   📦 Cache now has {cache_size_after} entries")

    # Load again (should read fresh from file)
    print("\n4. Loading lesson plans again (should be fresh)...")
    lessons2 = loader.load_lesson_plans("python", "functions")
    if lessons2:
        print(f"   ✅ Loaded {len(lessons2.get('lessons', {}))} lessons")
        first_lesson2 = list(lessons2.get("lessons", {}).values())[0]
        title2 = first_lesson2.get("title", "")
        print(f"   📖 First lesson title: {title2}")
        has_test2 = "test" in title2.lower()
        print(f"   🔍 Contains 'test': {has_test2}")
    else:
        print("   ❌ Failed to load lessons after cache clear")
        return

    # Test full cache clear
    print("\n5. Testing full cache clear...")
    loader.clear_cache()
    cache_size_final = len(loader._cache)
    print(f"   📦 Cache now has {cache_size_final} entries (should be 0)")

    print("\n🎉 Cache clearing test completed!")

    if has_test and has_test2:
        print("✅ SUCCESS: Your 'test' changes are visible in the JSON file!")
        print("💡 Now when you update lessons through the admin panel,")
        print("   the cache will be automatically cleared and changes will be visible.")
    else:
        print("⚠️  The 'test' changes might not be in the expected location.")


if __name__ == "__main__":
    test_cache_clearing()
