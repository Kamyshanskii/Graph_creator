from src.agents import answer_question


def main() -> None:
    print("Ассистент по matplotlib (DeepSeek + RAG + 2 агента, CLI).")
    print("Задавай вопросы про построение графиков, например:")
    print("  - Как построить гистограмму столбца age?")
    print("  - Как сделать boxplot value по category?")
    print("Чтобы выйти, напиши: exit / quit / выход.\n")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", "выход"}:
            print("Пока!")
            break

        result = answer_question(q)

        print("\n[Маршрутизатор]")
        print("Категория: {cat}, поисковый запрос: {sq}".format(
            cat=result["routing"]["category"],
            sq=result["routing"]["search_query"],
        ))

        print("\n[Ответ ассистента]\n")
        print(result["answer"])
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
