screen -dRRS connie-skipper bash -c "\
cd ~/connie-skipper;\
git pull;\
module load py-notebook/6.1.5-gcc-8.3.0-lim66g5 softwares/texlive/2017;\
jupyter --no-browser --port=8889;\
"
