screen -dRRS connie-skipper bash -c "
echo 'screen started.'
cd ~/connie-skipper
echo -e '\e[33mupdate connie-skipper.git\e[0m'
git pull
PATH=\"\"
module purge
module load py-notebook softwares/texlive python/3.6.3-gcc-8.3.0
if [ ! -x ~/.local/bin/pip3 ]
then 
    echo -e '\e[33install pip\e[0m'
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
fi
PATH=~/.local/bin:\$PATH
echo \$PATH
if ! python3 -c 'import xarray, scipy'
then 
    echo -e '\e[33minstall requirements:\e[0m'
    pip3 install --upgrade pip --user
    pip3 install matplotlib --user
    pip3 install astropy --user
    pip3 install ipympl --user
    pip3 install voila --user
    pip3 install scipy --user    
    pip3 install ipywidgets --user
    pip3 install xarray --user
fi
echo -e '\e[33mstart jupyter notebook.\e[0m'
voila skipper.ipynb --no-browser --port=8889
"
