codedir = ../../code/
figdir = figs/

MAIN = pres

# FIGS = si_vef_s2sa si vef solconv_homo solconv_reed dl_it dl_err mfemgrid lldggrid
FIGS = vef_flow vef_discretized
FIGS := $(addsuffix .pdf, $(FIGS))
FIGS := $(addprefix $(figdir), $(FIGS))

# TABS = mms
TABS := $(addsuffix .tex, $(TABS))
TABS := $(addprefix $(figdir), $(TABS))

LD = $(codedir)ld.py

# generate all figures 
.PHONY : all 
all : $(figdir) $(FIGS) $(MAIN).pdf

$(figdir) :
	mkdir $(figdir) 

# algorithm flow chart figure 
$(figdir)vef_flow.pdf : ink/vef_flow.svg 
	# generate pdfs 
	inkscape -D -z --file=$< --export-pdf=$@ --export-latex
	# make it recognize directory structure 
	sed -i 's/$(notdir $@)/figs\/$(notdir $@)/' $@_tex
	# delete lines with page=2 
	sed -i '/page=2/d' $@_tex

# colored version 
$(figdir)vef_discretized.pdf : ink/vef_discretized.svg 
	# generate pdfs 
	inkscape -D -z --file=$< --export-pdf=$@ --export-latex
	# make it recognize directory structure 
	sed -i 's/$(notdir $@)/figs\/$(notdir $@)/' $@_tex
	# delete lines with page=2 
	sed -i '/page=2/d' $@_tex

# compile latex 
$(MAIN).pdf: $(FIGS) $(TABS) $(MAIN).tex
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
	