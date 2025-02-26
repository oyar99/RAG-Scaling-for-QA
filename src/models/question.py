class Question(dict):
    def __init__(self, question_id: str, question: str, conversation_id: str):
        dict.__init__(self, question_id=question_id, question=question, conversation_id=conversation_id)

    def __repr__(self):
        return f"Question(question_id={self.question_id}, question={self.question}), conversation_id={self.conversation_id}"