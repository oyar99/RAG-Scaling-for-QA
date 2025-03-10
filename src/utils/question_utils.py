"""Utility functions for questions processing."""

from models.question_answer import QuestionAnswer, QuestionCategory


def filter_questions(
    questions: list[QuestionAnswer],
    limit: int = None,
    category: int = None
) -> list[QuestionAnswer]:
    """
    Filters the list of questions based on the specified category and limit.
    If no category is specified, all questions are returned up to the limit.

    Args:
        questions (list[QuestionAnswer]): the list of questions
        limit (int, optional): the maximum number of questions to be returned
        category (int, optional): the category to be returned. All if not specified

    Returns:
        filtered_questions (list[QuestionAnswer]): the filtered list of questions
    """
    filtered_questions = [
        question for question in questions
        if (category is None and category != QuestionCategory.ADVERSARIAL) or int(question['category']) == category
    ]

    return filtered_questions[:limit]
