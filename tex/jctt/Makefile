codedir = ../../code/
figdir = figs/

PYTHON = python3
MPLRC = matplotlibrc 
BIB = references.bib

MAIN = jctt
FIGS = solconv converge acceleration
# FIGS = si_vef_s2sa si vef solconv_homo solconv_reed dl_it dl_err mfemgrid lldggrid
FIGS := $(addsuffix .pdf, $(FIGS))
FIGS := $(addprefix $(figdir), $(FIGS))

INK = lldg mfem 
INK := $(addsuffix .pdf, $(INK))
INK := $(addprefix $(figdir), $(INK))

TABS = mms perm_dl
TABS := $(addsuffix .tex, $(TABS))
TABS := $(addprefix $(figdir), $(TABS))

LD = $(codedir)ld.py

VPATH = ink

# generate all figures 
.PHONY : all 
all : $(figdir) $(FIGS) $(INK) $(TABS) $(MAIN).pdf

# make figure directory 
$(figdir) : 
	-mkdir $(figdir) 

# build tables 
$(figdir)%.tex : %.py
	$(PYTHON) $< $@ 

# built figures 
$(figdir)%.pdf : %.py $(MPLRC) 
	$(PYTHON) $< $@ 

# build inkscape figures 
$(figdir)%.pdf : %.svg
	# generate pdfs 
	inkscape -D -z --file=$< --export-pdf=$@ --export-latex
	# make it recognize directory structure 
	sed -i 's/$(notdir $@)/figs\/$(notdir $@)/' $@_tex
	# delete lines with page=2 
	sed -i '/page=2/d' $@_tex

# compile latex 
$(MAIN).pdf: $(MAIN).tex $(FIGS) $(TABS) $(BIB)
	latexmk -pdf \
		-pdflatex="pdflatex --interaction=nonstopmode --shell-escape %O %S" $(MAIN)

# remove tex auxilary files 
.PHONY : clean 
clean : 
	rm -f $(MAIN).aux 
	rm -f $(MAIN).bbl 
	rm -f $(MAIN).blg
	rm -f $(MAIN).fdb_latexmk
	rm -f $(MAIN).fls
	rm -f $(MAIN).log
	rm -f $(MAIN).nav
	rm -f $(MAIN).out
	rm -f $(MAIN).snm
	rm -f $(MAIN).toc
	rm -f $(MAIN).synctex.gz
	rm -f $(MAIN).fff
	rm -f $(MAIN).ttt
	