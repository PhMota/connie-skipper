echo "forward che port 8889 to local port 8889"
ssh -N -f -L localhost:8889:localhost:8889 che.cbpf.br -p 13900
echo "update remote app and start jupyter at che in port 8889"
ssh -t che.cbpf.br -p 13900 "
source ~/.bashrc
source ~/.bash_profile
echo 'screen started.'
if [ ! -d '~/connie-skipper' ] 
then
    git clone https://github.com/PhMota/connie-skipper.git
fi
bash ~/connie-skipper/run.sh
"