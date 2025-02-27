"""A module to create a question class."""
class Question(dict):
    """A question class that inherits from dict.

    Args:
        dict: inherits from dict
    """

    def __init__(self, question_id: str, question: str, conversation_id: str):
        dict.__init__(self, question_id=question_id,
                      question=question, conversation_id=conversation_id)

    def __repr__(self):
        return f"""Question(question_id={self.get('question_id')}, question={self.get('question')}),\
    conversation_id={self.get('conversation_id')}"""
