pdflatex initialization_hv.tex
pdflatex pareto_fronts.tex

pdfcrop initialization_hv.pdf
pdfcrop pareto_fronts.pdf

rm *.log
rm *.aux
rm *.fdb_latexmk
rm *.fls
rm *.synctex.gz