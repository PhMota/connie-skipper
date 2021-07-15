echo -e "\n" > runremote.log
echo -e "\e[34mwaiting for the server port...\e[0m"
bash -c "
tail -f runremote.log | grep http://localhost -q
PORT=\$(cat runremote.log | grep http://localhost | grep -o '[0-9]*')
ssh -N -f -L localhost:\$PORT:localhost:\$PORT che.cbpf.br -p 13900
echo -e '\\e[33mapp is running locally at the above link (open link in browser)\\e[0m'
echo -e '\\n'
echo -e \"\\r\\e[4mhttp://localhost:\$PORT/\\e[0m\"
" &

echo -e "\e[34mstarting server at che\e[0m"
ssh -t che.cbpf.br -p 13900 "
echo -e '\e[33mscreen started.\e[0m'
pwd
if [ ! -d ~/connie-skipper/ ] 
then
    echo 'cloning the repository from https://github.com/PhMota/connie-skipper.git'
    git clone https://github.com/PhMota/connie-skipper.git
else
    echo 'updating connie-skipper'
    cd ~/connie-skipper/
    git pull
fi
if ! bash ~/connie-skipper/run.sh
then
    exec bash
fi
" | tee runremote.log
echo -e "\e[33mclosing log file\e[0m"
killall tail
PORT=$(cat runremote.log | grep http://localhost | grep -o '[0-9]*')
echo -e "\e[33mclosing the port fowarding at $PORT\e[0m"
ssh -O cancel -L localhost:$PORT:localhost:$PORT che.cbpf.br -p 13900
