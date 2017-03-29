codedir = ../../code/
figdir = figs/

MAIN = pres
FIGS = $(figdir)diffLimit.pdf $(figdir)converge_una.pdf $(figdir)converge_acc.pdf \
	$(figdir)hlim.pdf $(figdir)ooa.pdf $(figdir)dvs.pdf $(figdir)edd.pdf $(figdir)corrected.pdf 

# generate all figures 
.PHONY : all 
all : $(figdir) $(FIGS) $(MAIN).pdf

$(figdir) : 
	mkdir $(figdir) 

$(figdir)diffLimit.pdf : $(codedir)diffLimit.py 
	$< $@

$(figdir)converge_una.pdf : $(codedir)converge.py 
	$< $(figdir)converge_una.pdf $(figdir)converge_acc.pdf 

$(figdir)converge_acc.pdf : $(codedir)converge.py 
	$< $(figdir)converge_una.pdf $(figdir)converge_acc.pdf 

$(figdir)hlim.pdf : $(codedir)hlimit.py 
	$< $@

$(figdir)ooa.pdf : $(codedir)mms.py 
	$< $@

$(figdir)dvs.pdf : $(codedir)dvs.py 
	$< $(figdir)dvs.pdf $(figdir)edd.pdf $(figdir)corrected.pdf 

$(figdir)edd.pdf : $(codedir)dvs.py 
	$< $(figdir)dvs.pdf $(figdir)edd.pdf $(figdir)corrected.pdf 

$(figdir)corrected.pdf : $(codedir)dvs.py 
	$< $(figdir)dvs.pdf $(figdir)edd.pdf $(figdir)corrected.pdf 

# compile latex 
$(MAIN).pdf: $(MAIN).tex $(FIGS)
	latexmk -pdf\
		-pdflatex="pdflatex --interaction=nonstopmode --shell-escape %O %S" $(MAIN)