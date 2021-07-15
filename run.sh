screen -dRRS connie-skipper bash -c "
echo 'screen started.'
cd ~/connie-skipper
echo -e '\e[33mupdate connie-skipper.git\e[0m'
git pull
module load py-notebook/6.1.5-gcc-8.3.0-lim66g5 softwares/texlive/2017 python/3.6.3-gcc-5.3.0
if [ ! -x ~/.local/bin/pip3 ]
then 
    echo -e '\e[33install pip\e[0m'
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
fi
if [ ! $(python3 -c "import xarray, matplotlib, astropy, ipympl, voila, ipywidgets" 2> /dev/null) ]
then 
    echo -e '\e[33minstall requirements:\e[0m'
    pip3 install --upgrade pip --user
    pip3 install matplotlib --user
    pip3 install astropy --user
    pip3 install ipympl --user
    pip3 install voila --user
    pip3 install ipywidgets --user
    pip3 install xarray --user
fi
echo -e '\e[33mstart jupyter notebook.\e[0m'
voila skipper.ipynb --no-browser --port=8889
"
