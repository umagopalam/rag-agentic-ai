from src.graph import graph


test_questions = [
    "What is Agentic AI?",
    "What are the key capabilities of Agentic AI?",
    "What are the pillars of Agentic AI?",
    "What are the building blocks of Agentic AI?",
    "What are multi-agent systems?",
    "Who won the 2022 FIFA World Cup?"
]


for question in test_questions:

    result = graph.invoke({
        "question": question,
        "context": [],
        "answer": "",
        "score": 0.0
    })

    print("=" * 80)
    print("QUESTION:", question)
    print("ANSWER:", result["answer"])
    print("RETRIEVAL SCORE:", result["score"])