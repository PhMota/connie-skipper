if [ $# -eq 0 ]
  then
    echo "please provide the remote folder into which to install the app, e.g.:"
    echo "bash runremote.sh '~/'"
    exit -1
fi

ssh che.cbpf.br -p 13900 install.sh