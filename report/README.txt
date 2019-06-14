Compile as:
latex conference
bibtex conference
latex conference
latex conference

Now you have a dvi file. Get a ps file as:
dvips -t letter -o conference.ps -Pdf -G0 conference.dvi

Now you have a ps file. Get a pdf as:
ps2pdf conference.ps