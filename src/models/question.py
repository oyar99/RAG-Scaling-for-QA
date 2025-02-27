"""A module to create a question class."""

from enum import Enum


class QuestionCategory(Enum):
    """An enum to represent the question category.

    Args:
        Enum: inherits from Enum
    """
    MULTI_HOP = 1
    TEMPORAL = 2
    OPEN_DOMAIN = 3
    SINGLE_HOP = 4
    ADVERSARIAL = 5

class Question(dict):
    """A question class that inherits from dict.

    Args:
        dict: inherits from dict
    """

    def __init__(self, question_id: str, question: str, conversation_id: str, category: QuestionCategory):
        dict.__init__(self, question_id=question_id,
                      question=question, conversation_id=conversation_id, category=category)

    def __repr__(self):
        return f"""Question(question_id={self.get('question_id')}, question={self.get('question')}),\
    conversation_id={self.get('conversation_id')}, category={self.get('category')})"""

