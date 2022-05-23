pdflatex mapping.tex
pdflatex alg_flowchart.tex
pdflatex preprocessing.tex

pdfcrop mapping.pdf
pdfcrop alg_flowchart.pdf
pdfcrop preprocessing.pdf

rm *.log
rm *.aux
rm *.fdb_latexmk
rm *.fls
rm *.synctex.gz