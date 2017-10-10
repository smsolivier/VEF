codedir = ../../code/
figdir = figs/
SRC = src/
MPL = matplotlibrc

MAIN = pres

FIGS = si_vef_s2sa si vef solconv_homo dl_it dl_err reed_solution
FIGS := $(addsuffix .pdf, $(FIGS))
FIGS := $(addprefix $(figdir), $(FIGS))

INK = vef_flow vef_discretized lldg mfem scattering naive slopes \
	scattering_color naive_color average_color slopes_color interpolate_color \
	MFEM_grid_color
INK := $(addsuffix .pdf, $(INK))
INK := $(addprefix $(figdir), $(INK))

TABS = mms
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
$(figdir)si_vef_s2sa.pdf : $(SRC)acceleration.py $(MPL)
	./$< $@ 

# plot convergence of phi and edd for VEF and SI 
$(figdir)si.pdf : $(SRC)converge.py $(MPL)
	./$< $@ $(figdir)vef.pdf 

# create table of MMS errors 
$(figdir)mms.tex : $(SRC)mms.py
	./$< $@ 

# plot of solution convergence for homogeneous and Reed's problem 
$(figdir)solconv_homo.pdf : $(SRC)solconv.py $(MPL)
	./$< $@ $(figdir)solconv_reed.pdf 

# diff limit of 4 permutations 
$(figdir)dl_it.pdf : $(SRC)perm_dl.py $(MPL)
	./$< $@ $(figdir)dl_err.pdf 

# reed solution 
$(figdir)reed_solution.pdf : $(SRC)reed.py $(MPL)
	./$< $@

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
	rm -f berkeley-eps-converted-to.pdf
