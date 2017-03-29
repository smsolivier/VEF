codedir = ../../code/
figdir = figs/

MAIN = ans
FIGS = $(figdir)eddCon_si.pdf $(figdir)eddCon_mu.pdf $(figdir)accel.pdf 

# generate all figures 
.PHONY : all
all : $(figdir) $(FIGS) $(MAIN).pdf

# make figure directory 
$(figdir) : 
	mkdir $(figdir)

# generate convergence plots 
$(figdir)eddCon_si.pdf : $(codedir)converge.py 
	$(codedir)converge.py $(figdir)eddCon_si.pdf $(figdir)eddCon_mu.pdf 

$(figdir)eddCon_mu.pdf : $(codedir)converge.py 
	$(codedir)converge.py $(figdir)eddCon_si.pdf $(figdir)eddCon_mu.pdf 

# generate accel figure 
$(figdir)accel.pdf : $(codedir)checkAccel.py 
	$(codedir)checkAccel.py $@

$(MAIN).pdf: $(MAIN).tex $(FIGS)
	latexmk -pdf\
		-pdflatex="pdflatex --interaction=nonstopmode --shell-escape %O %S" $(MAIN)