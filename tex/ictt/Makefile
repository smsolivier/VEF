codedir = ../../code/
figdir = figs/

MAIN = pres

FIGS = si_vef_s2sa si vef solconv_homo solconv_reed dl_it dl_err 
FIGS := $(addsuffix .pdf, $(FIGS))
FIGS := $(addprefix $(figdir), $(FIGS))

INK = vef_flow vef_discretized lldg mfem scattering naive slopes average \
	scattering_color naive_color
INK := $(addsuffix .pdf, $(INK))
INK := $(addprefix $(figdir), $(INK))

# TABS = mms
TABS := $(addsuffix .tex, $(TABS))
TABS := $(addprefix $(figdir), $(TABS))

LD = $(codedir)ld.py

# generate all figures 
.PHONY : all 
all : $(figdir) $(INK) $(MAIN).pdf

$(figdir) :
	mkdir $(figdir) 

$(figdir)%.pdf : ink/%.svg
	# generate pdfs 
	inkscape -D -z --file=$< --export-pdf=$@ --export-latex
	# make it recognize directory structure 
	sed -i 's/$(notdir $@)/figs\/$(notdir $@)/' $@_tex
	# delete lines with page=2 
	sed -i '/page=2/d' $@_tex

# plot SI v VEF v S2SA
$(figdir)si_vef_s2sa.pdf : acceleration.py 
	./$< $@ 

# plot convergence of phi and edd for VEF and SI 
$(figdir)si.pdf : converge.py 
	./$< $@ $(figdir)vef.pdf 

# create table of MMS errors 
$(figdir)mms.tex : mms.py
	./$< $@ 

# plot of solution convergence for homogeneous and Reed's problem 
$(figdir)solconv_homo.pdf : solconv.py 
	./$< $@ $(figdir)solconv_reed.pdf 

# diff limit of 4 permutations 
$(figdir)dl_it.pdf : perm_dl.py 
	./$< $@ $(figdir)dl_err.pdf 

# compile latex 
$(MAIN).pdf: $(FIGS) $(TABS) $(INK) $(MAIN).tex
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
	rm -f $(MAIN).pdf	