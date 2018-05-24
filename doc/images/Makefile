SOURCES := $(shell find $(SOURCEDIR) -name '*.tex')
OBJS := $(SOURCES:%.tex=%.png)

%.png: %.pdf
	convert -density 300 $< -quality 90 $@

%.pdf: %.tex
	pdflatex $<

.PHONY: all
all: $(OBJS)
	rm -rf *.pdf *.out *.aux *.log
