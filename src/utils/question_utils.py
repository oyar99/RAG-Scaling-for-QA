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
    filtered_questions = questions

    if category is not None:
        filtered_questions = [
            question for question in questions
            if int(question['category']) == category
        ]
    else:
        # Remove unanswerable questions
        filtered_questions = [
            question for question in questions
            if question['category'] != QuestionCategory.ADVERSARIAL
        ]

    # Dedupe questions by question_id
    filtered_questions = list(
        {q['question_id']: q for q in filtered_questions}.values())

    return filtered_questions[:limit]
