all: $(patsubst %.tex,%.pdf,$(wildcard *.tex))

.PRECIOUS: %.pdf

% : %.pdf
	@echo -n ""

%.dvi: %.tex
	latex $<
	latex $<

%.ps: %.dvi
	dvips -t letter -Ppdf -G0 $< -o $@

%.pdf: %.ps
	ps2pdf $<

clean:
	rm -rf *.bbl *.blg *.aux *.log *~ *.bak *.ps *.dvi *.log *.out *.tmp 

cleanall:
	rm -rf *.bbl *.blg *.aux *.log *~ *.bak *.ps *.dvi *.log *.pdf svnver.tex *.out *.tmp 

spell:
	ispell -f ispell.words -t *.tex
