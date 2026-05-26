TEMPLATE_QUESTION = """You are an expert in natural language processing. Your task is to generate a clear and concise question about what or who is associated with subject "{subject}" through the relation "{relation}".
    An example for this subject and relation would be: "{question}".
    Requirements:
    - Be creative to not use exactly the same phrasing each time.
    - The question should suited for multiple years adding a {{year}} indicator for the user to change it.
    Output Format:
    Question: <your generated question here>."""

TEMPLATE_PH_QUESTION = """You are an expert in natural language processing. Your task is to reformulate if necessary the following question making sure that "{subject}" still appears: "In {{year}}, who held the position of {subject}?".
    Output Format:
    Question: <your generated question here>."""

TEMPLATE_AR_QUESTION = """You are an expert in natural language processing. Your task is to reformulate if necessary the following question making sure that "{subject}" still appears: "In {{year}}, who received the {subject} ?".
    Output Format:
    Question: <your generated question here>."""

TEMPLATE_DISTRACTOR = """You are an expert in natural language processing and logic puzzles, skilled at generating plausible yet misleading distractor options that challenge users to distinguish between correct and incorrect answers. Your task is to create a distractor answer for the following question that is plausible but incorrect, ensuring it does not match the correct answer provided.
    Question: {question} | Existing Answers: {answer}
    Requirements:
    - The distractor must be plausible and relevant to the question.
    - It should not be the same as the existing answer.
    - Ensure that the distractor is distinct from the existing answer.
    Output Format:
    Distractor: <your distractor answer here>."""

PREDEFINED_TEMPLATE = {
    "head of government": "Who was the head of government of {subject} in {{year}}?",
    "head of state": "Who was the head of state of {subject} in {{year}}?",
    "position held": "Who held the position of {subject} in {{year}}?",
    "award received": "Who received the {subject} in {{year}}?",
    "winner": "What did {subject} win in {{year}}?",
    "member of sports team": "Which sports team did {subject} play for in {{year}}?",
    "head coach": "Who was the head coach of {subject} in {{year}}?",
    "league or competition": "In which league or competition did {subject} compete in {{year}}?",
    "chairperson": "Who served as the chairperson of {subject} in {{year}}?",
    "overall winner general classification": "Who was the overall winner of the general classification for the {subject} in {{year}}?",
    "second overall": "Who finished second overall in the {subject} in {{year}}?",
    "third overall": "Who finished third overall in the {subject} in {{year}}?",
    "winner of the points classification": "Who won the points classification in the {subject} in {{year}}?",
    "winner of the mountain classification": "Who won the mountain classification in the {subject} in {{year}}?",
    "winner of the young rider classification": "Who won the young rider classification in the {subject} in {{year}}?",
    "winner of the teams classification by time": "Which team won the teams classification by time in the {subject} in {{year}}?",
    "winner of the most combative rider": "Who was named the most combative rider in the {subject} in {{year}}?",
    "employer": "Who employed {subject} in {{year}}?",
    "member of political party": "Which political party was {subject} affiliated with in {{year}}?",
    "educated at": "Where did {subject} receive their education in {{year}}?",
    "owned by": "Who owned {subject} in {{year}}?",
    "stage winner": "Who won the stage in the {subject} in {{year}}?",
    "leader of the young rider classification": "Who was leading the young rider classification in the {subject} in {{year}}?",
    "leader of the points classification": "Who was leading the points classification in the {subject} in {{year}}?",
    "leader of the mountain classification": "Who was leading the mountain classification in the {subject} in {{year}}?",
    "leader of the teams classification by time": "Which team was leading the teams classification by time in the {subject} in {{year}}?",
    "most combative rider": "Who was considered the most combative rider in the {subject} in {{year}}?",
    "overall leader at the end of the stage": "Who was the overall leader at the end of the stage in the {subject} in {{year}}?",
    "winner of the sprint classification": "Who won the sprint classification in the {subject} in {{year}}?",
}
