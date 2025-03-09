"""A module to create a question-answer class."""
from enum import IntEnum

from models.document import Document


class QuestionCategory(IntEnum):
    """An enum to represent the question category.

    Args:
        Enum: inherits from Enum
    """
    MULTI_HOP = 1
    TEMPORAL = 2
    OPEN_DOMAIN = 3
    SINGLE_HOP = 4
    ADVERSARIAL = 5

class QuestionAnswer(dict):
    """A question class that inherits from dict.

    Args:
        dict: inherits from dict
    """

    # pylint: disable-next=too-many-positional-arguments,too-many-arguments
    def __init__(self, question_id: str, question: str, answer: str, category: QuestionCategory, docs: list[Document]):
        dict.__init__(self, question_id=question_id,
                      question=question, answer=answer, category=category, docs=docs)

    def __repr__(self):
        return f"""Question(question_id={self.get('question_id')}, question={self.get('question')}),\
    answer={self.get('answer')}, category={self.get('category')}), docs={self.get('docs')}"""
