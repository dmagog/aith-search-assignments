from __future__ import annotations

from pathlib import Path

import markdown


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = PROJECT_ROOT / "report.md"
REPORT_HTML = PROJECT_ROOT / "report.html"


CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 980px;
  margin: 40px auto;
  padding: 0 24px 64px;
  line-height: 1.6;
  color: #1f2328;
  background: #ffffff;
}
h1, h2, h3, h4 { line-height: 1.25; margin-top: 1.6em; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 16px 0 24px;
  font-size: 14px;
}
th, td {
  border: 1px solid #d0d7de;
  padding: 8px 10px;
  text-align: left;
}
th { background: #f6f8fa; }
code {
  background: #f6f8fa;
  padding: 0.15em 0.35em;
  border-radius: 6px;
}
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 18px auto 28px;
  border: 1px solid #d0d7de;
  border-radius: 8px;
}
ul, ol { margin-bottom: 16px; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def main() -> None:
    text = REPORT_MD.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ДЗ2 Report</title>
  <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(REPORT_HTML)


if __name__ == "__main__":
    main()
