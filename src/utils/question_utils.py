"""Utility functions for the questions processing."""

from models.question_answer import QuestionAnswer, QuestionCategory


def filter_questions(questions: list[QuestionAnswer], limit: int = None, category: int = None) -> list:
    """Filters the questions based on the category and limit.

    Args:
        questions (list): the list of questions
        limit (int, optional): the limit of questions to be returned
        category (int): the category to be returned. All if not specified

    Returns:
        list: the filtered list of questions
    """
    filtered_questions = [
        question for question in questions
        if (category is None and category != QuestionCategory.ADVERSARIAL) or int(question['category']) == category
    ]

    return filtered_questions[:limit]
