pdflatex large_scale_hv.tex
pdflatex large_scale_times.tex
lualatex pareto_fronts.tex

pdfcrop large_scale_hv.pdf
pdfcrop large_scale_times.pdf
pdfcrop pareto_fronts.pdf

rm *.log
rm *.aux
rm *.fdb_latexmk
rm *.fls
rm *.synctex.gz