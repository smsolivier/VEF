codedir = ../../code/
figdir = figs/

MAIN = jctt

FIGS = si_vef_s2sa si vef
FIGS := $(addsuffix .pdf, $(FIGS))
FIGS := $(addprefix $(figdir), $(FIGS))

TABS = mms
TABS := $(addsuffix .tex, $(TABS))
TABS := $(addprefix $(figdir), $(TABS))

# generate all figures 
.PHONY : all 
all : $(figdir) $(FIGS) $(TABS) $(MAIN).pdf

# make figure directory 
$(figdir) : 
	-mkdir $(figdir) 

# plot SI v VEF v S2SA
$(figdir)si_vef_s2sa.pdf : acceleration.py 
	./$< $@ 

# plot convergence of phi and edd for VEF and SI 
$(figdir)si.pdf : converge.py 
	./$< $@ $(figdir)vef.pdf 

# create table of MMS errors 
$(figdir)mms.tex : mms.py 
	./$< $@ 

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
	