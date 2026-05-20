# DZ0: Search Engine Evaluation Criteria

В папке лежит короткий отчет по формализации критериев оценки веб-поисковиков.

## С чего читать

- [report.pdf](report.pdf) — собранный PDF-отчёт.
- [report_source.md](report_source.md) — текстовая версия отчёта.
- [report.tex](report.tex) — LaTeX-исходник.

## Что внутри

- [references.bib](references.bib) — библиография.
- [profile_scores.png](profile_scores.png) — график профильных оценок.
- [acl.sty](acl.sty), [acl_natbib.bst](acl_natbib.bst) — локальные файлы шаблона.

Сборка:

```bash
cd dz0
xelatex report.tex
bibtex report
xelatex report.tex
xelatex report.tex
```
