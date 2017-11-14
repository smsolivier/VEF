codedir = ../../code/
figdir = figs/

PYTHON = python3
MPLRC = matplotlibrc 

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

# # plot SI v VEF v S2SA
# $(figdir)si_vef_s2sa.pdf : acceleration.py 
# 	./$< $@ 

# # plot convergence of phi and edd for VEF and SI 
# $(figdir)si.pdf : converge.py 
# 	./$< $@ $(figdir)vef.pdf 

# # create table of MMS errors 
# $(figdir)mms.tex : mms.py
# 	./$< $@ 

# # plot of solution convergence for homogeneous and Reed's problem 
# $(figdir)solconv_homo.pdf : solconv.py 
# 	./$< $@ $(figdir)solconv_reed.pdf 

# # diff limit of 4 permutations 
# $(figdir)dl_it.pdf : perm_dl.py 
# 	./$< $@ $(figdir)dl_err.pdf 

# # latex SVG of MFEM figure 
# $(figdir)mfemgrid.pdf : ink/mfem.svg 
# 	# generate pdfs 
# 	inkscape -D -z --file=$< --export-pdf=$@ --export-latex
# 	# make it recognize directory structure 
# 	sed -i 's/mfemgrid.pdf/figs\/mfemgrid.pdf/' $(figdir)mfemgrid.pdf_tex 
# 	# delete lines with page=2 
# 	sed -i '/page=2/d' $(figdir)mfemgrid.pdf_tex 

# # latex SVG of LLDG figure 
# $(figdir)lldggrid.pdf : ink/lldg.svg 
# 	# generate pdfs 
# 	inkscape -D -z --file=$< --export-pdf=$@ --export-latex
# 	# make it recognize directory structure 
# 	sed -i 's/lldggrid.pdf/figs\/lldggrid.pdf/' $(figdir)lldggrid.pdf_tex 
# 	# delete lines with page=2 
# 	sed -i '/page=2/d' $(figdir)lldggrid.pdf_tex 

$(figdir)%.tex : %.py
	$(PYTHON) $< $@ 

$(figdir)%.pdf : %.py $(MPLRC) 
	$(PYTHON) $< $@ 

$(figdir)%.pdf : %.svg
	# generate pdfs 
	inkscape -D -z --file=$< --export-pdf=$@ --export-latex
	# make it recognize directory structure 
	sed -i 's/$(notdir $@)/figs\/$(notdir $@)/' $@_tex
	# delete lines with page=2 
	sed -i '/page=2/d' $@_tex

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
	