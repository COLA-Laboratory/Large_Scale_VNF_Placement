pdflatex dcell_hv.tex
pdflatex dcell_times.tex
pdflatex fat_tree_hv.tex
pdflatex fat_tree_times.tex
pdflatex leaf_spine_hv.tex
pdflatex leaf_spine_times.tex
pdflatex pareto_fronts.tex

pdfcrop dcell_hv.pdf
pdfcrop dcell_times.pdf
pdfcrop fat_tree_hv.pdf
pdfcrop fat_tree_times.pdf
pdfcrop leaf_spine_hv.pdf
pdfcrop leaf_spine_times.pdf
pdfcrop pareto_fronts.pdf

rm *.log
rm *.aux
rm *.fdb_latexmk
rm *.fls
rm *.synctex.gz