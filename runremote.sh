echo "update remote app and start jupyter at che in port 8889"
touch runremote.log
tail -f runremote.log | grep http://localhost: | grep -o "[0-9]*" -q;\
PORT = $(tail runremote.log | grep http://localhost: | grep -o "[0-9]*");\
ssh -N -f -L localhost:$PORT:localhost:$PORT che.cbpf.br -p 13900;\
echo "app running at (open link in browser)";\
echo "http://localhost:$PORT/" &

ssh -t che.cbpf.br -p 13900 "
source ~/.bashrc
source ~/.bash_profile
echo 'screen started.'
if [ ! -d '~/connie-skipper' ] 
then
    git clone https://github.com/PhMota/connie-skipper.git
else
    git pull
fi
bash ~/connie-skipper/run.sh
" | tee runremote.log