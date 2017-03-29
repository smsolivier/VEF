# store where pythonscripts are located in relation to makefile 
codedir = ../../code/ 
# where to store figures 
figdir = figs/

# name of main tex file 
MAIN = ans
# list of figures 
FIGS = $(figdir)eddCon_si.pdf $(figdir)eddCon_mu.pdf $(figdir)accel.pdf 

# generate all figures 
.PHONY : all
all : $(figdir) $(FIGS) $(MAIN).pdf

# make figure directory 
$(figdir) : 
	mkdir $@

# generate convergence plots 
$(figdir)eddCon_si.pdf : $(codedir)converge.py 
	$< $(figdir)eddCon_si.pdf $(figdir)eddCon_mu.pdf 

$(figdir)eddCon_mu.pdf : $(codedir)converge.py 
	$< $(figdir)eddCon_si.pdf $(figdir)eddCon_mu.pdf 

# generate accel figure 
$(figdir)accel.pdf : $(codedir)checkAccel.py 
	$< $@

# compile latex 
$(MAIN).pdf: $(MAIN).tex $(FIGS)
	latexmk -pdf\
		-pdflatex="pdflatex --interaction=nonstopmode --shell-escape %O %S" $(MAIN)