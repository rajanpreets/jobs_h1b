# from langgraph.graph import END, StateGraph

# from chains.question_relevance import question_relevance
# from chains.document_relevance import document_relevance
# from nodes import evaluate, generate_answer, retrieve, search_online
# from state import GraphState



# def any_doc_irrelevant(state):
#     if state["online_search"]:
#         return "Search Online"
#     else:
#         return "Generate Answer"


# def hallucinations(state: GraphState) -> str:
#     question = state["question"]
#     documents = state["documents"]
#     solution = state["solution"]

#     score = document_relevance.invoke(
#         {"documents": documents, "solution": solution}
#     )

#     if score.binary_score:
#         score = question_relevance.invoke({"question": question, "solution": solution})
#         if score.binary_score:
#             return "Answers Question"
#         else:
#             return "Question not addressed"
#     else:
#         return "Hallucinations detected"


# workflow = StateGraph(GraphState)

# workflow.add_node("retrievedocuments", retrieve)
# workflow.add_node("Grade Documents", evaluate)
# workflow.add_node("Generate Answer", generate_answer)
# workflow.add_node("Search Online", search_online)

# workflow.set_entry_point("Retrieve Documents")
# workflow.add_edge("Retrieve Documents", "Grade Documents")
# workflow.add_conditional_edges(
#     "Grade Documents",
#     any_doc_irrelevant,
#     {
#         "Search Online": "Search Online",
#         "Generate Answer": "Generate Answer",
#     },
# )

# workflow.add_conditional_edges(
#     "Generate Answer",
#     hallucinations,
#     {
#         "Hallucinations detected": "Generate Answer",
#         "Answers Question": END,
#         "Question not addressed": "Search Online",
#     },
# )
# workflow.add_edge("Search Online", "Generate Answer")
# workflow.add_edge("Generate Answer", END)

# graph = workflow.compile()

# graph.get_graph().draw_mermaid_png(output_file_path="graph.png")