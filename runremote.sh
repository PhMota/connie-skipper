if [ $# -eq 0 ]
  then
    echo "please provide the remote folder into which to install the app, e.g.:"
    echo "bash runremote.sh '~/'"
    exit -1
fi

echo "forward che port 8889 to local port 8889"
ssh -N -f -L localhost:8889:localhost:8889 che.cbpf.br -p 13900
echo "update remote app and start jupyter at che in port 8889"
ssh che.cbpf.br -p 13900 ""