codedir = ../../code/
figdir = figs/

MAIN = pres

# FIGS = si_vef_s2sa si vef solconv_homo solconv_reed dl_it dl_err mfemgrid lldggrid
FIGS = vef_flow
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

# latex SVG of MFEM figure 
$(figdir)mfemgrid.pdf : ink/mfem.svg 
	# generate pdfs 
	inkscape -D -z --file=$< --export-pdf=$@ --export-latex
	# make it recognize directory structure 
	sed -i 's/mfemgrid.pdf/figs\/mfemgrid.pdf/' $(figdir)mfemgrid.pdf_tex 
	# delete lines with page=2 
	sed -i '/page=2/d' $(figdir)mfemgrid.pdf_tex 

# latex SVG of LLDG figure 
$(figdir)lldggrid.pdf : ink/lldg.svg 
	# generate pdfs 
	inkscape -D -z --file=$< --export-pdf=$@ --export-latex
	# make it recognize directory structure 
	sed -i 's/lldggrid.pdf/figs\/lldggrid.pdf/' $(figdir)lldggrid.pdf_tex 
	# delete lines with page=2 
	sed -i '/page=2/d' $(figdir)lldggrid.pdf_tex 

# algorithm flow chart figure 
$(figdir)vef_flow.pdf : ink/vef_flow.svg 
	# generate pdfs 
	inkscape -D -z --file=$< --export-pdf=$@ --export-latex
	# make it recognize directory structure 
	sed -i 's/vef_flow.pdf/figs\/vef_flow.pdf/' $(figdir)vef_flow.pdf_tex
	# delete lines with page=2 
	sed -i '/page=2/d' $(figdir)vef_flow.pdf_tex

# compile latex 
$(MAIN).pdf: $(MAIN).tex $(FIGS) $(TABS)
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
	