"""
DataLoader utility for dynamically loading subject and subtopic data from JSON files.
Handles error cases and provides caching for performance.
"""

import json
import os
from typing import Dict, List, Optional, Any
from flask import current_app


class DataLoader:
    """Handles loading of subject and subtopic data from JSON files."""

    def __init__(self, data_root_path: str):
        """
        Initialize the DataLoader with the root data path.

        Args:
            data_root_path: Path to the data directory (e.g., "/path/to/data")
        """
        self.data_root = data_root_path
        self._cache = {}

    def _load_json_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a JSON file and return its contents.

        Args:
            file_path: Absolute path to the JSON file

        Returns:
            Dictionary containing JSON data, or None if file doesn't exist or is corrupted
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            if current_app:
                current_app.logger.error(f"JSON file not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            if current_app:
                current_app.logger.error(f"Invalid JSON in file {file_path}: {e}")
            return None
        except Exception as e:
            if current_app:
                current_app.logger.error(f"Error loading JSON file {file_path}: {e}")
            return None

    def _get_cache_key(
        self, subject: str, subtopic: str = None, file_type: str = None
    ) -> str:
        """Generate a cache key for storing loaded data."""
        if subtopic and file_type:
            return f"{subject}_{subtopic}_{file_type}"
        elif file_type:
            return f"{subject}_{file_type}"
        else:
            return subject

    def load_subject_config(self, subject: str) -> Optional[Dict[str, Any]]:
        """
        Load subject configuration (keywords, settings, etc.).

        Args:
            subject: Subject name (e.g., "python")

        Returns:
            Dictionary containing subject config, or None if not found
        """
        cache_key = self._get_cache_key(subject, file_type="config")

        if cache_key in self._cache:
            return self._cache[cache_key]

        config_path = os.path.join(
            self.data_root, "subjects", subject, "subject_config.json"
        )
        config_data = self._load_json_file(config_path)

        if config_data:
            self._cache[cache_key] = config_data

        return config_data

    def load_subject_info(self, subject: str) -> Optional[Dict[str, Any]]:
        """
        Load subject information (name, description, icon, etc.).

        Args:
            subject: Subject name (e.g., "python")

        Returns:
            Dictionary containing subject info, or None if not found
        """
        cache_key = self._get_cache_key(subject, file_type="info")

        if cache_key in self._cache:
            return self._cache[cache_key]

        info_path = os.path.join(
            self.data_root, "subjects", subject, "subject_info.json"
        )
        info_data = self._load_json_file(info_path)

        if info_data:
            self._cache[cache_key] = info_data

        return info_data

    def load_quiz_data(self, subject: str, subtopic: str) -> Optional[Dict[str, Any]]:
        """
        Load quiz data for a specific subject/subtopic.

        Args:
            subject: Subject name (e.g., "python")
            subtopic: Subtopic name (e.g., "functions")

        Returns:
            Dictionary containing quiz data, or None if not found
        """
        cache_key = self._get_cache_key(subject, subtopic, "quiz")

        if cache_key in self._cache:
            return self._cache[cache_key]

        quiz_path = os.path.join(
            self.data_root, "subjects", subject, subtopic, "quiz_data.json"
        )
        quiz_data = self._load_json_file(quiz_path)

        if quiz_data:
            self._cache[cache_key] = quiz_data

        return quiz_data

    def load_question_pool(
        self, subject: str, subtopic: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load question pool for remedial quizzes.

        Args:
            subject: Subject name (e.g., "python")
            subtopic: Subtopic name (e.g., "functions")

        Returns:
            Dictionary containing question pool, or None if not found
        """
        cache_key = self._get_cache_key(subject, subtopic, "questions")

        if cache_key in self._cache:
            return self._cache[cache_key]

        pool_path = os.path.join(
            self.data_root, "subjects", subject, subtopic, "question_pool.json"
        )
        pool_data = self._load_json_file(pool_path)

        if pool_data:
            self._cache[cache_key] = pool_data

        return pool_data

    def load_lesson_plans(
        self, subject: str, subtopic: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load lesson plans for a subject/subtopic.

        Args:
            subject: Subject name (e.g., "python")
            subtopic: Subtopic name (e.g., "functions")

        Returns:
            Dictionary containing lesson plans, or None if not found
        """
        cache_key = self._get_cache_key(subject, subtopic, "lessons")

        if cache_key in self._cache:
            return self._cache[cache_key]

        lessons_path = os.path.join(
            self.data_root, "subjects", subject, subtopic, "lesson_plans.json"
        )
        lessons_data = self._load_json_file(lessons_path)

        if lessons_data:
            self._cache[cache_key] = lessons_data

        return lessons_data

    def load_videos(self, subject: str, subtopic: str) -> Optional[Dict[str, Any]]:
        """
        Load video data for a subject/subtopic.

        Args:
            subject: Subject name (e.g., "python")
            subtopic: Subtopic name (e.g., "functions")

        Returns:
            Dictionary containing video data, or None if not found
        """
        cache_key = self._get_cache_key(subject, subtopic, "videos")

        if cache_key in self._cache:
            return self._cache[cache_key]

        videos_path = os.path.join(
            self.data_root, "subjects", subject, subtopic, "videos.json"
        )
        videos_data = self._load_json_file(videos_path)

        if videos_data:
            self._cache[cache_key] = videos_data

        return videos_data

    def get_subject_keywords(self, subject: str) -> List[str]:
        """
        Get allowed AI analysis tags for a subject.
        
        Note: Function name kept for backwards compatibility, but now returns tags.

        Args:
            subject: Subject name (e.g., "python")

        Returns:
            List of allowed tags, empty list if not found
        """
        config = self.load_subject_config(subject)
        if config:
            # Support both old and new format during migration
            tags = config.get("allowed_tags", config.get("allowed_keywords", []))
            # Ensure all tags are lowercase
            return [tag.lower() for tag in tags]
        return []

    def get_quiz_questions(self, subject: str, subtopic: str) -> List[Dict[str, Any]]:
        """
        Get quiz questions for a subject/subtopic.

        Args:
            subject: Subject name (e.g., "python")
            subtopic: Subtopic name (e.g., "functions")

        Returns:
            List of question dictionaries, empty list if not found
        """
        quiz_data = self.load_quiz_data(subject, subtopic)
        if quiz_data:
            return quiz_data.get("questions", [])
        return []

    def get_question_pool_questions(
        self, subject: str, subtopic: str
    ) -> List[Dict[str, Any]]:
        """
        Get question pool for remedial quizzes.

        Args:
            subject: Subject name (e.g., "python")
            subtopic: Subtopic name (e.g., "functions")

        Returns:
            List of question dictionaries, empty list if not found
        """
        pool_data = self.load_question_pool(subject, subtopic)
        if pool_data:
            return pool_data.get("questions", [])
        return []

    def get_quiz_title(self, subject: str, subtopic: str) -> str:
        """
        Get the title for a quiz.

        Args:
            subject: Subject name (e.g., "python")
            subtopic: Subtopic name (e.g., "functions")

        Returns:
            Quiz title string, or a default title if not found
        """
        quiz_data = self.load_quiz_data(subject, subtopic)
        if quiz_data:
            return quiz_data.get(
                "quiz_title", f"{subject.title()} {subtopic.title()} Quiz"
            )
        return f"{subject.title()} {subtopic.title()} Quiz"

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()

    def clear_cache_for_subject_subtopic(self, subject: str, subtopic: str):
        """
        Clear cache entries for a specific subject/subtopic combination.

        Args:
            subject: Subject name (e.g., "python")
            subtopic: Subtopic name (e.g., "functions")
        """
        # Clear all cache entries for this subject/subtopic
        cache_keys_to_remove = []
        for key in self._cache.keys():
            if key.startswith(f"{subject}_{subtopic}_"):
                cache_keys_to_remove.append(key)

        for key in cache_keys_to_remove:
            del self._cache[key]

    def validate_subject_subtopic(self, subject: str, subtopic: str) -> bool:
        """
        Check if a subject/subtopic combination exists.

        Args:
            subject: Subject name (e.g., "python")
            subtopic: Subtopic name (e.g., "functions")

        Returns:
            True if the combination exists, False otherwise
        """
        # Check if the directory structure exists
        subtopic_path = os.path.join(self.data_root, "subjects", subject, subtopic)
        if not os.path.exists(subtopic_path):
            return False

        # Check if at least quiz_data.json exists
        quiz_path = os.path.join(subtopic_path, "quiz_data.json")
        return os.path.exists(quiz_path)

    def find_lessons_by_tags(
        self, subject: str, target_tags: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Find lessons that match any of the target tags.

        Args:
            subject: Subject name (e.g., "python")
            target_tags: List of tags to search for

        Returns:
            List of matching lessons with metadata
        """
        matching_lessons = []

        try:
            # Get all subtopics for the subject
            subject_config = self.load_subject_config(subject)
            if not subject_config or "subtopics" not in subject_config:
                return matching_lessons

            # Search through all subtopics
            for subtopic_id in subject_config["subtopics"].keys():
                lesson_plans = self.load_lesson_plans(subject, subtopic_id)
                if not lesson_plans or "lessons" not in lesson_plans:
                    continue

                # Check each lesson in the subtopic
                for lesson_id, lesson_data in lesson_plans["lessons"].items():
                    lesson_tags = set(lesson_data.get("tags", []))
                    target_tags_set = set(target_tags)

                    # If any lesson tags match target tags
                    if not lesson_tags.isdisjoint(target_tags_set):
                        matching_lessons.append(
                            {
                                "subject": subject,
                                "subtopic": subtopic_id,
                                "lesson_id": lesson_id,
                                "title": lesson_data.get("title", ""),
                                "tags": lesson_data.get("tags", []),
                                "matching_tags": list(
                                    lesson_tags.intersection(target_tags_set)
                                ),
                            }
                        )

        except Exception as e:
            current_app.logger.error(f"Error finding lessons by tags: {e}")

        return matching_lessons

    def discover_subjects(self) -> Dict[str, Dict[str, Any]]:
        """
        Auto-discover subjects by scanning the subjects directory for folders
        containing subject_info.json files.

        Returns:
            Dictionary of subjects in the same format as subjects.json
        """
        subjects = {}
        subjects_dir = os.path.join(self.data_root, "subjects")
        
        if not os.path.exists(subjects_dir):
            if current_app:
                current_app.logger.warning(f"Subjects directory not found: {subjects_dir}")
            return subjects

        try:
            # Scan all directories in the subjects folder
            for item in os.listdir(subjects_dir):
                subject_path = os.path.join(subjects_dir, item)
                
                # Skip if not a directory
                if not os.path.isdir(subject_path):
                    continue
                
                # Look for subject_info.json
                subject_info_path = os.path.join(subject_path, "subject_info.json")
                subject_config_path = os.path.join(subject_path, "subject_config.json")
                
                # Subject must have both files to be valid
                if os.path.exists(subject_info_path) and os.path.exists(subject_config_path):
                    subject_info = self._load_json_file(subject_info_path)
                    subject_config = self._load_json_file(subject_config_path)
                    
                    if subject_info and subject_config:
                        # Calculate subtopic count
                        subtopic_count = len(subject_config.get("subtopics", {}))
                        
                        # Merge info and add calculated fields
                        subjects[item] = {
                            **subject_info,
                            "subtopic_count": subtopic_count,
                            "status": subject_info.get("status", "active"),
                            "created_date": subject_info.get("created_date", "2025-01-01")
                        }
                        
                        if current_app:
                            current_app.logger.info(f"Discovered subject: {item} - {subject_info.get('name', 'Unknown')}")
                    else:
                        if current_app:
                            current_app.logger.warning(f"Invalid subject files in directory: {item}")
                else:
                    if current_app:
                        current_app.logger.debug(f"Skipping directory (missing required files): {item}")
                        
        except Exception as e:
            if current_app:
                current_app.logger.error(f"Error discovering subjects: {e}")
        
        return subjects

    def migrate_tags_for_subject(self, subject: str) -> bool:
        """
        Migrate tags from keywords to tags format and collect existing tags from content.
        
        Args:
            subject: Subject name to migrate
            
        Returns:
            True if migration was successful
        """
        try:
            subject_config_path = os.path.join(self.data_root, "subjects", subject, "subject_config.json")
            
            if not os.path.exists(subject_config_path):
                return False
                
            # Load current config
            with open(subject_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Collect existing tags from various sources
            all_tags = set()
            
            # Add existing allowed_keywords
            existing_keywords = config.get("allowed_keywords", [])
            all_tags.update([tag.lower() for tag in existing_keywords])
            
            # Add existing allowed_tags if any
            existing_tags = config.get("allowed_tags", [])
            all_tags.update([tag.lower() for tag in existing_tags])
            
            # Scan all subtopics for lesson and question tags
            subject_dir = os.path.join(self.data_root, "subjects", subject)
            if os.path.exists(subject_dir):
                for item in os.listdir(subject_dir):
                    item_path = os.path.join(subject_dir, item)
                    if os.path.isdir(item_path) and item not in ["__pycache__"]:
                        # Check lesson plans
                        lesson_plans_path = os.path.join(item_path, "lesson_plans.json")
                        if os.path.exists(lesson_plans_path):
                            try:
                                with open(lesson_plans_path, 'r', encoding='utf-8') as f:
                                    lesson_data = json.load(f)
                                    lessons = lesson_data.get("lessons", {})
                                    for lesson_id, lesson_content in lessons.items():
                                        lesson_tags = lesson_content.get("tags", [])
                                        all_tags.update([tag.lower() for tag in lesson_tags])
                            except Exception as e:
                                if current_app:
                                    current_app.logger.warning(f"Error reading lesson plans for {subject}/{item}: {e}")
                        
                        # Check quiz data
                        quiz_data_path = os.path.join(item_path, "quiz_data.json")
                        if os.path.exists(quiz_data_path):
                            try:
                                with open(quiz_data_path, 'r', encoding='utf-8') as f:
                                    quiz_data = json.load(f)
                                    questions = quiz_data.get("questions", [])
                                    for question in questions:
                                        question_tags = question.get("tags", [])
                                        all_tags.update([tag.lower() for tag in question_tags])
                            except Exception as e:
                                if current_app:
                                    current_app.logger.warning(f"Error reading quiz data for {subject}/{item}: {e}")
                        
                        # Check question pool
                        pool_data_path = os.path.join(item_path, "question_pool.json")
                        if os.path.exists(pool_data_path):
                            try:
                                with open(pool_data_path, 'r', encoding='utf-8') as f:
                                    pool_data = json.load(f)
                                    questions = pool_data.get("questions", [])
                                    for question in questions:
                                        question_tags = question.get("tags", [])
                                        all_tags.update([tag.lower() for tag in question_tags])
                            except Exception as e:
                                if current_app:
                                    current_app.logger.warning(f"Error reading question pool for {subject}/{item}: {e}")
            
            # Update config with new format
            config["allowed_tags"] = sorted(list(all_tags))
            
            # Remove old allowed_keywords if it exists
            if "allowed_keywords" in config:
                del config["allowed_keywords"]
            
            # Save updated config
            with open(subject_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            if current_app:
                current_app.logger.info(f"Migrated {len(all_tags)} tags for subject '{subject}': {sorted(list(all_tags))}")
            
            return True
            
        except Exception as e:
            if current_app:
                current_app.logger.error(f"Error migrating tags for subject {subject}: {e}")
            return False

    def migrate_all_subjects_tags(self) -> Dict[str, bool]:
        """
        Migrate tags for all discovered subjects.
        
        Returns:
            Dictionary mapping subject names to migration success status
        """
        results = {}
        subjects = self.discover_subjects()
        
        for subject_id in subjects.keys():
            results[subject_id] = self.migrate_tags_for_subject(subject_id)
            
        return results
