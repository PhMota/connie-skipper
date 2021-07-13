screen -dRRS connie-skipper bash -c "\
cd $(cat PATH.txt);\
git pull;\
module load py-notebook/6.1.5-gcc-8.3.0-lim66g5 softwares/texlive/2017;\
pip3 install --upgrade pip3 --user;\
pip3 install ipywidgets --user;\
pip3 install xarray --user;\
jupyter notebook --no-browser --port=8889;\
"
