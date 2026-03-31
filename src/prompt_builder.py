def build_prompt(table_text, question, language="ko"):
    if language == "ko":
        instruction = (
            "아래 표를 보고 질문에 답하세요.\n"
            "표의 정보만 사용하세요.\n"
            "설명 없이 최종 답만 출력하세요.\n"
        )
    else:
        instruction = (
            "Read the table and answer the question.\n"
            "Use only the information in the table.\n"
            "Output only the final answer without explanation.\n"
        )

    return f"{instruction}\n[Table]\n{table_text}\n\n[Question]\n{question}\n\n[Answer]"