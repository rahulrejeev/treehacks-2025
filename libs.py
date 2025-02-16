from dataclasses import dataclass
from typing import List, Tuple
from collections import namedtuple

QuestionDataWrapper = namedtuple('QuestionDataWrapper', ["session_id", "question", "agg_feedback"])

@dataclass
class MeditationSession:
    questions_responses: List[QuestionDataWrapper]
    session_id: str

@dataclass
class User:
    user_id: str
    sessions: List[MeditationSession]