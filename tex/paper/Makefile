codedir = ../../code/
figdir = figs/

MAIN = jctt
FIGS = diffLimit diffError mms hlimit
FIGS := $(addsuffix .pdf, $(FIGS))
FIGS := $(addprefix $(figdir), $(FIGS))

# generate all figures 
.PHONY : all 
all : $(figdir) $(FIGS) $(MAIN).pdf

# make figure directory 
$(figdir) : 
	-mkdir $(figdir) 

# generate diffusion limit test 
$(figdir)diffLimit.pdf : perm_dl.py 
	./$< $@ $(figdir)diffError.pdf 

$(figdir)diffError.pdf : perm_dl.py 
	./$< $(figdir)diffLimit.pdf $@

# generate mms plot 
$(figdir)mms.pdf : mms.py 
	./$< $@

# generate hlimit plot 
$(figdir)hlimit.pdf : hlimit.py 
	./$< $@ 

# compile latex 
$(MAIN).pdf: $(MAIN).tex $(FIGS)
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
	