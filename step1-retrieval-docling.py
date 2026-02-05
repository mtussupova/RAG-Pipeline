"""
Step 1: Data Extraction & Validation
Извлечение данных из PDF с помощью Docling и валидация качества через Visual LLM
"""

import json
import base64
from pathlib import Path

# Загрузка переменных окружения из .env
from dotenv import load_dotenv
load_dotenv()

# PDF processing
from docling.document_converter import DocumentConverter
import fitz  # PyMuPDF для создания скриншотов страниц

# LLM для валидации (используем OpenAI с vision)
from openai import OpenAI


# Конфигурация
PDF_PATH = "kaztelecom.pdf"
PAGES_TO_EXTRACT = [2, 3]  # Страницы для извлечения (1-indexed)
OUTPUT_DIR = Path("extraction_output")


def create_page_screenshot(pdf_path: str, page_num: int, dpi: int = 150) -> bytes:
    """
    Создаёт скриншот страницы PDF в формате PNG.

    Args:
        pdf_path: Путь к PDF файлу
        page_num: Номер страницы (1-indexed)
        dpi: Разрешение изображения

    Returns:
        PNG изображение в байтах
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # fitz использует 0-indexed

    # Создаём матрицу масштабирования для нужного DPI
    zoom = dpi / 72  # 72 - стандартный DPI для PDF
    matrix = fitz.Matrix(zoom, zoom)

    # Рендерим страницу в изображение
    pix = page.get_pixmap(matrix=matrix)
    png_bytes = pix.tobytes("png")

    doc.close()
    return png_bytes


def extract_text_with_docling(pdf_path: str, pages: list[int]) -> dict[int, str]:
    """
    Извлекает текст из указанных страниц PDF с помощью Docling.

    Args:
        pdf_path: Путь к PDF файлу
        pages: Список номеров страниц (1-indexed)

    Returns:
        Словарь {номер_страницы: извлечённый_текст}
    """
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    # Получаем полный markdown
    full_markdown = doc.export_to_markdown()

    # Docling не всегда разделяет по страницам, поэтому извлекаем весь документ
    # и возвращаем для каждой страницы (в реальном сценарии нужна более точная логика)
    extracted = {}

    # Пробуем получить текст по страницам через итерацию по элементам
    for page_num in pages:
        page_content = []
        for element, _level in doc.iterate_items():
            # Проверяем, относится ли элемент к нужной странице
            if hasattr(element, 'prov') and element.prov:
                for prov in element.prov:
                    if hasattr(prov, 'page_no') and prov.page_no == page_num:
                        if hasattr(element, 'text') and element.text:
                            page_content.append(element.text)
                        elif hasattr(element, 'export_to_markdown'):
                            try:
                                page_content.append(element.export_to_markdown(doc))
                            except TypeError:
                                # Fallback для элементов без метода или с другой сигнатурой
                                pass
                        break

        if page_content:
            extracted[page_num] = "\n\n".join(page_content)
        else:
            # Если не удалось разделить, возвращаем весь документ
            extracted[page_num] = full_markdown

    return extracted


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Кодирует изображение в base64."""
    return base64.b64encode(image_bytes).decode("utf-8")


def validate_extraction_with_llm(
    screenshot_base64: str,
    extracted_text: str,
    client: OpenAI
) -> dict:
    """
    Валидирует качество извлечения с помощью Visual LLM.

    Args:
        screenshot_base64: Скриншот страницы в base64
        extracted_text: Извлечённый текст/markdown
        client: OpenAI клиент

    Returns:
        Словарь с оценками качества
    """
    prompt = """Тебе даны:
1. Скриншот оригинальной страницы PDF
2. Извлечённый текст/Markdown из этой страницы

Оцени от 1 до 5 качество извлечения по следующим критериям:

**Структура (1-5):**
- Сохранена ли структура заголовков (H1, H2, H3)?
- Правильно ли распознаны списки (нумерованные, маркированные)?

**Таблицы (1-5):**
- Сохранена ли структура таблиц?
- Читаемы ли данные в ячейках?
- Правильно ли выровнены столбцы и строки?

**Форматирование (1-5):**
- Сохранены ли жирные, курсивные элементы?
- Правильно ли обработаны формулы/спецсимволы?

**Полнота (1-5):**
- Весь ли текст извлечён?
- Нет ли пропущенных блоков?

**Итоговая оценка:** (среднее значение)

Ответ дай ТОЛЬКО в формате JSON без дополнительного текста:
{
  "structure_score": X,
  "tables_score": X,
  "formatting_score": X,
  "completeness_score": X,
  "overall_score": X,
  "comments": "..."
}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Извлечённый текст:\n\n{extracted_text}"
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    # Парсим JSON из ответа
    response_text = response.choices[0].message.content.strip()

    # Убираем возможные markdown блоки
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]

    return json.loads(response_text)


def main():
    """Основная функция для извлечения и валидации данных."""

    # Создаём директорию для выходных данных
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Step 1: Data Extraction & Validation")
    print("=" * 60)

    # Инициализируем OpenAI клиент
    client = OpenAI()

    # Извлекаем текст с помощью Docling
    print(f"\n📄 Извлечение текста из {PDF_PATH} (страницы {PAGES_TO_EXTRACT})...")
    extracted_texts = extract_text_with_docling(PDF_PATH, PAGES_TO_EXTRACT)

    # Подготавливаем test cases
    test_cases = []

    for page_num in PAGES_TO_EXTRACT:
        print(f"\n📸 Создание скриншота страницы {page_num}...")
        screenshot_bytes = create_page_screenshot(PDF_PATH, page_num)

        # Сохраняем скриншот
        screenshot_path = OUTPUT_DIR / f"page_{page_num}.png"
        with open(screenshot_path, "wb") as f:
            f.write(screenshot_bytes)
        print(f"   Сохранено: {screenshot_path}")

        # Сохраняем извлечённый текст
        text_path = OUTPUT_DIR / f"page_{page_num}_extracted.md"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(extracted_texts.get(page_num, ""))
        print(f"   Извлечённый текст: {text_path}")

        test_cases.append({
            "page_num": page_num,
            "pdf_page_screenshot": str(screenshot_path),
            "extracted_text": extracted_texts.get(page_num, ""),
            "extraction_method": "docling"
        })

    # Валидация с помощью LLM
    print("\n" + "=" * 60)
    print("🤖 Валидация качества извлечения (LLM as a Judge)")
    print("=" * 60)

    results = []

    for case in test_cases:
        page_num = case["page_num"]
        print(f"\n📊 Оценка страницы {page_num}...")

        # Читаем скриншот и кодируем в base64
        with open(case["pdf_page_screenshot"], "rb") as f:
            screenshot_base64 = encode_image_to_base64(f.read())

        # Получаем оценку от LLM
        try:
            evaluation = validate_extraction_with_llm(
                screenshot_base64,
                case["extracted_text"],
                client
            )

            result = {
                "page_num": page_num,
                "extraction_method": case["extraction_method"],
                "evaluation": evaluation
            }
            results.append(result)

            # Выводим результаты
            print(f"   Структура:     {evaluation.get('structure_score', 'N/A')}/5")
            print(f"   Таблицы:       {evaluation.get('tables_score', 'N/A')}/5")
            print(f"   Форматирование:{evaluation.get('formatting_score', 'N/A')}/5")
            print(f"   Полнота:       {evaluation.get('completeness_score', 'N/A')}/5")
            print(f"   Итого:         {evaluation.get('overall_score', 'N/A')}/5")
            print(f"   Комментарии:   {evaluation.get('comments', 'N/A')}")

        except Exception as e:
            print(f"   ❌ Ошибка при оценке: {e}")
            results.append({
                "page_num": page_num,
                "extraction_method": case["extraction_method"],
                "error": str(e)
            })

    # Сохраняем результаты
    results_path = OUTPUT_DIR / "validation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Результаты сохранены: {results_path}")

    # Итоговая статистика
    print("\n" + "=" * 60)
    print("📈 Итоговая статистика")
    print("=" * 60)

    successful_evals = [r for r in results if "evaluation" in r]
    if successful_evals:
        avg_overall = sum(r["evaluation"]["overall_score"] for r in successful_evals) / len(successful_evals)
        print(f"\nСредняя итоговая оценка: {avg_overall:.2f}/5")

    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
