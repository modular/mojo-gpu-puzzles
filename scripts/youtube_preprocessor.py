import json
import re
import sys


def process_content(content):
    pattern = r"\{\{\s*youtube\s+(\S+)(?:\s+(\S+))?\s*\}\}"

    def replace(match):
        video_id = match.group(1)
        size_class = match.group(2) or ""
        return f"""<div class="video-container {size_class}">
<iframe src="https://www.youtube.com/embed/{video_id}" allowfullscreen></iframe>
</div>"""

    return re.sub(pattern, replace, content)


def process_items(items):
    for item in items:
        if "Chapter" in item:
            chapter = item["Chapter"]
            chapter["content"] = process_content(chapter["content"])
            process_items(chapter.get("sub_items", []))


def main():
    # Handle mdbook's "supports" arg by indicating html support
    if len(sys.argv) > 1 and sys.argv[1] == "supports":
        renderer = sys.argv[2] if len(sys.argv) > 2 else ""
        sys.exit(0 if renderer == "html" else 1)

    context, book = json.load(sys.stdin)
    for section in book["sections"]:
        if "Chapter" in section:
            chapter = section["Chapter"]
            chapter["content"] = process_content(chapter["content"])
            process_items(chapter.get("sub_items", []))

    print(json.dumps(book))


if __name__ == "__main__":
    main()
