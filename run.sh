screen -dRRS connie-skipper bash -c "
echo 'screen started.'
cd ~/connie-skipper
module unload softwares/python/2.7-gnu-5.3
module load py-notebook/6.1.5-gcc-8.3.0-lim66g5 softwares/texlive/2017 python/3.6.3-gcc-5.3.0
pwd
echo 'modules loaded:'
module list
echo 'update connie-skipper.git'
git pull
python3 --version
python3 -m pip --version
if [ ! -x ~/.local/bin/pip3 ]
then 
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
fi
pip3 --version
echo 'update requirements:'
pip3 --version
pip3 install --upgrade pip3 --user
pip3 install voila --user
pip3 install ipywidgets --user
pip3 install xarray --user
echo 'start jupyter notebook'
voila skipper.ipynb --no-browser --port=8889
"
