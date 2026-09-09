main.pdf: main.tex sections/*.tex references.bib $(wildcard figures/*.pdf figures/*/*.pdf figures/*.png figures/*/*.png)
	pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null; pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main | tail -3
clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
