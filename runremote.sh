echo "forward che port 8889 to local port 8889"
ssh -N -f -L localhost:8889:localhost:8889 che.cbpf.br -p 13900
echo "update remote app and start jupyter at che in port 8889"
ssh che.cbpf.br -p 13900 "bash ~/connie-skipper/run.sh"