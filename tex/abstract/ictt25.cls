\LoadClass[11pt,twoside]{article}
\typeout{Loading ICTT25 document class: includes [11pt,twoside]{article}}
\typeout{adapted by Nick Gentile, who got it from Vittorio Romano (UNICT) who got the ICTT23 template from Jim Warsa (LANL) who got the most recent MC2013 template form 
         R.C. Martineau at INL from A. Sood at LANL, from J. Wagner (ORNL) 
         who obtained the original class file by Jim Warsa, LANL, 16 July 2002}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bibliography
\RequirePackage{citesort}

\renewcommand{\topskip}{2em}

\def\@maketitle{%
  \newpage
  \vspace*{-\topskip}      % remove the initial space
  \begingroup\centering    % instead of \begin{center}
  \let \footnote \thanks
  \hrule height \z@        % to avoid the insertion of lineskip glue
    {\Large \@title \par}%
     \vspace*{-\topskip}      % remove the initial space
%    \vskip 1.5em 
    {\large
      \lineskip .5em 
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em 
    {\large \@date}%
  \par\endgroup            % instead of \end{center}
%  \vskip 1.5em             % <--- modify this to adjust the separation
%  \vspace*{-\topskip}      % remove the initial space
}

%
% rename reference section
\renewcommand{\refname}{References}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% page layout stuff
\RequirePackage{setspace}
\RequirePackage{indentfirst}
\RequirePackage{fancyhdr}
\RequirePackage{afterpage}
\RequirePackage{times}
\RequirePackage{authblk}

\setlength{\textwidth}{6.5in}
\setlength{\textheight}{9.0in}
\setlength{\marginparsep}{0pt}
\setlength{\marginparwidth}{0pt}
\setlength{\oddsidemargin}{0pt}
\setlength{\evensidemargin}{0pt}
\setlength{\hoffset}{0pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% headers and such (fancyhdr)
%
\setlength{\voffset}{0pt}
\setlength{\hoffset}{0.0in}
\setlength{\topmargin}{0pt}
%\setlength{\topmargin}{16pt}
\setlength{\headsep}{16.8pt}
\setlength{\headheight}{14pt}
\setlength{\footskip}{0.4in}

\setlength{\floatsep}{6pt plus 3pt minus 1pt}
\setlength{\textfloatsep}{6pt plus 3pt minus 1pt}
%\setlength{\intextsep}{4pt plus 1pt minus 1pt}
%\setlength{\abovedisplayskip}{4pt plus 1pt minus 1pt}
%\setlength{\belowdisplayskip}{0.5\baselineskip plus 1pt minus 1pt}

\setlength{\parindent}{0pt}
\addtolength{\parskip}{1.0\baselineskip}

\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\g@addto@macro{\maketitle}{\thispagestyle{fancy}\vspace{-4em}}
\date{}

\lhead{\footnotesize{%
\emph{25th International Conference on Transport Theory,
Monterey, California, 16-20 October 2017}}}%
\lfoot{}

\pagestyle{fancy}

\def\section{\@startsection{section}{1}{\z@}
{1.0ex plus 1ex minus .2ex}{.2ex plus .2ex}{\normalsize\bf\center}}
\def\subsection{\@startsection{subsection}{2}{\z@} 
{1.0ex plus 1ex minus .2ex}{.2ex plus .2ex}{\normalsize\bf}}
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}
{1.0ex plus 1ex minus .2ex}{.2ex plus .2ex}{\normalsize\bf}}

\renewcommand{\thetable}{\@Roman\c@table}

\newcommand{\Section}[1]{\section{\hskip -1em. #1}}
\newcommand{\Subsection}[1]{\subsection{\hskip -1em. #1}}
\newcommand{\Subsubsection}[1]{\subsubsection{\hskip -1em. #1}}

\renewcommand{\thefootnote}{\fnsymbol{footnote}}
\renewcommand{\footnotesize}{\fontsize{9}{10.8pt}\selectfont}
\renewcommand{\small}{\fontsize{10}{12pt}\selectfont}
\renewcommand{\large}{\fontsize{14}{16.8pt}\selectfont}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% borrowed from pagecounting.sty 
% http://www-users.york.ac.uk/~zrs1/Software/pagecounting.sty)

% History:
%  Created 2000/04/07
% -------- Identification
%
%\NeedsTeXFormat{LaTeX2e}
%\ProvidesPackage{pagecounting}[2000/04/07 <Zoe Stephenson>]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcounter{l@stpage}
\newcommand{\storel@st}{\immediate\write\@mainaux{%
\setcounter{l@stpage}{\thepage}%
% also could use: \setcounter{l@stpage}{\arabic{page}}%
% also could use: \setcounter{l@stpage}{\number{page}}%
\addtocounter{l@stpage}{0}}%  Adjust due to page counter post-advance
}
\renewcommand{\thel@stpage}{\arabic{l@stpage}}
\AtEndDocument{\storel@st}
\newcommand{\totalpages}{\thel@stpage}
\newcommand{\lastpage}{\value{l@stpage}}
