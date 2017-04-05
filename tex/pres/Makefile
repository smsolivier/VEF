codedir = ../../code/
figdir = figs/

MAIN = pres
FIGS = diffLimit converge_una converge_acc hlim ooa dvs edd exSol
FIGS := $(addsuffix .pdf, $(FIGS))
FIGS := $(addprefix $(figdir), $(FIGS))

# generate all figures 
.PHONY : all 
all : $(figdir) $(FIGS) $(MAIN).pdf 
	
$(figdir) : 
	-mkdir $(figdir) 

$(figdir)diffLimit.pdf : diffLimit.py 
	./$< $@

$(figdir)converge_una.pdf : converge.py 
	./$< $(figdir)converge_una.pdf $(figdir)converge_acc.pdf 

$(figdir)converge_acc.pdf : converge.py 
	./$< $(figdir)converge_una.pdf $(figdir)converge_acc.pdf 

$(figdir)hlim.pdf : hlimit.py 
	./$< $@

$(figdir)ooa.pdf : mms.py 
	./$< $@

$(figdir)dvs.pdf : $(codedir)dvs.py 
	$< $(figdir)dvs.pdf $(figdir)edd.pdf $(figdir)corrected.pdf 

$(figdir)edd.pdf : $(codedir)dvs.py 
	$< $(figdir)dvs.pdf $(figdir)edd.pdf $(figdir)corrected.pdf 

$(figdir)corrected.pdf : $(codedir)dvs.py 
	$< $(figdir)dvs.pdf $(figdir)edd.pdf $(figdir)corrected.pdf 

$(figdir)exSol.pdf : exSol.py 
	./$< $@ 

# compile latex 
$(MAIN).pdf: $(MAIN).tex $(FIGS)
	latexmk -pdf \
		-pdflatex="pdflatex --interaction=nonstopmode --shell-escape %O %S" $(MAIN) > /dev/null

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